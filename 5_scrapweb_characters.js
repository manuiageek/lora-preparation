import puppeteer from "puppeteer";
import fs from "fs";
import path from "path";
import fetch from "node-fetch"; // installer avec "npm install node-fetch" si nécessaire
import { createInterface } from "readline";

// Fonction utilitaire pour poser une question à l'utilisateur
function askQuestion(query) {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) =>
    rl.question(query, (ans) => {
      rl.close();
      resolve(ans);
    })
  );
}

// Fonction utilitaire pour télécharger un fichier depuis une URL vers un chemin local
async function downloadImage(url, filePath) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Erreur lors du téléchargement ${url} - ${res.statusText}`);
  }
  const fileStream = fs.createWriteStream(filePath);
  return new Promise((resolve, reject) => {
    res.body.pipe(fileStream);
    res.body.on("error", (err) => {
      reject(err);
    });
    fileStream.on("finish", function () {
      resolve();
    });
  });
}

(async () => {
  // Récupère l'argument passé en ligne de commande si disponible
  let urlInput = process.argv[2];
  if (!urlInput) {
    // Si aucun argument n'est passé, on demande interactivement l'URL
    urlInput = await askQuestion("Veuillez entrer l'URL Character MyAnimeList à scraper : ");
  }
  let url = urlInput.trim();
  if (!url) {
    console.error("Aucune URL saisie. Fin du script.");
    process.exit(1);
  }

  // Vérifie si l'URL se termine par "/characters" ou "/characters/". Sinon, l'ajoute.
  if (!url.match(/\/characters\/?$/)) {
    url = url.replace(/\/+$/, ""); // Supprime les slashs en fin d'URL si présents
    url += "/characters";
  }

  // Extraction du dossier de base à partir de l'URL
  const urlObj = new URL(url);
  // Exemple de pathname : "/anime/40128/Arte/characters"
  const parts = urlObj.pathname.split("/").filter(Boolean); // ["anime", "40128", "Arte", "characters"]
  // Supposons que le nom à utiliser est le troisième segment (index 2) s'il est disponible
  const baseFolderName = parts.length >= 3 ? parts[2] : "default";
  // Dossier de destination pour les images (note : "images" est remplacé par "images_web")
  const baseDirPath = path.join("images_web", baseFolderName);
  // Assurez-vous que le dossier de base existe
  fs.mkdirSync(baseDirPath, { recursive: true });
  console.log(`Dossier de base créé ou existant : ${baseDirPath}`);

  // Lance le navigateur en mode headless
  const browser = await puppeteer.launch({
    headless: true,
    ignoreHTTPSErrors: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();

  // Définit un User-Agent réaliste
  await page.setUserAgent(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
  );

  try {
    // Accès à la page et attente des éléments contenant les noms de personnages
    await page.goto(url, { waitUntil: "networkidle2" });
    await page.waitForSelector("h3.h3_character_name", { timeout: 10000 });
    console.log("La page des personnages est chargée, début de l'extraction des liens...");

    // Extraction de tous les liens vers les pages de détails des personnages
    const characterLinks = await page.evaluate(() => {
      const elements = Array.from(
        document.querySelectorAll("h3.h3_character_name")
      );
      return elements
        .map((el) => (el.parentElement ? el.parentElement.href : null))
        .filter((link) => link !== null);
    });
    console.log(`Nombre de personnages trouvés : ${characterLinks.length}`);

    const results = [];
    const totalCharacters = characterLinks.length;

    // Itération sur chaque lien de personnage
    for (let i = 0; i < totalCharacters; i++) {
      const link = characterLinks[i];
      console.log(`Traitement du personnage ${i + 1}/${totalCharacters} : ${link}`);

      // Extraction du nom (dernier segment de l'URL)
      const urlName = link.split("/").pop();
      // Nettoyage du nom pour enlever les caractères indésirables
      const name = urlName.replace(/[^a-zA-Z0-9-_]/g, "");
      // Chemin complet du dossier du personnage dans la hiérarchie /images_web/<baseFolderName>/<name>
      const dirPath = path.join(baseDirPath, name);

      // Si le dossier existe et qu'il contient déjà au moins un fichier, on passe ce personnage
      if (fs.existsSync(dirPath) && fs.readdirSync(dirPath).length > 0) {
        console.log(`Le personnage ${name} a déjà été traité (dossier non vide). Passage au suivant.`);
        continue;
      }

      // Création (ou réinitialisation) du dossier pour ce personnage dans le dossier de base
      fs.mkdirSync(dirPath, { recursive: true });
      console.log(`Répertoire créé : ${dirPath}`);

      // Navigation vers la page de détail du personnage
      await page.goto(link, { waitUntil: "networkidle2" });

      let imageAvailable = true;
      // Vérifie s'il y a une image (timeout augmenté à 30000ms)
      try {
        await page.waitForSelector("img.portrait-225x350", { timeout: 30000 });
      } catch (imgError) {
        console.log(`Image non disponible pour ${name}.`);
        imageAvailable = false;
      }

      let imageUrl = "";
      // Si l'image est disponible, extraire son URL
      if (imageAvailable) {
        const details = await page.evaluate(() => {
          const imgElement = document.querySelector("img.portrait-225x350");
          const imageUrl = imgElement
            ? imgElement.getAttribute("data-src") ||
              imgElement.getAttribute("src") ||
              ""
            : "";
          return { imageUrl };
        });
        imageUrl = details.imageUrl;
        console.log("Nom extrait :", name);
        console.log("URL de l'image :", imageUrl);

        if (imageUrl) {
          // Détermination du nom de fichier à partir de l'URL de l'image (suppression des paramètres éventuels)
          const fileName = path.basename(imageUrl.split("?")[0]);
          const filePath = path.join(dirPath, fileName);

          try {
            console.log(`Téléchargement de l'image vers ${filePath}`);
            await downloadImage(imageUrl, filePath);
            console.log("Téléchargement réussi !");
          } catch (downloadError) {
            console.error("Erreur lors du téléchargement de l'image :", downloadError);
          }
        } else {
          console.log(`Aucune URL d'image trouvée pour ${name}.`);
        }
      } else {
        console.log(`Aucune image à télécharger pour ${name}.`);
      }

      results.push({ url: link, name, imageUrl: imageUrl || null });

      // Retour à la page principale pour continuer l'extraction
      await page.goto(url, { waitUntil: "networkidle2" });
      await page.waitForSelector("h3.h3_character_name", { timeout: 10000 });
    }

    console.log("Tous les personnages extraits :", results);
  } catch (error) {
    console.error("Erreur lors du scraping :", error);
  } finally {
    await browser.close();
  }
})();