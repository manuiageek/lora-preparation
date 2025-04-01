import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';  // installer avec "npm install node-fetch" si nécessaire

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
  // URL de la page listant les personnages
  const url = 'https://myanimelist.net/anime/53802/25-jigen_no_Ririsa/characters';

  // Lance le navigateur en mode headless
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Définit un User-Agent réaliste
  await page.setUserAgent(
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
  );

  try {
    // Accès à la page et attente des éléments contenant les noms de personnages
    await page.goto(url, { waitUntil: 'networkidle2' });
    await page.waitForSelector('h3.h3_character_name', { timeout: 10000 });
    console.log("La page des personnages est chargée, début de l'extraction des liens...");

    // Extraction de tous les liens vers les pages de détails des personnages
    const characterLinks = await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('h3.h3_character_name'));
      return elements.map(el => el.parentElement ? el.parentElement.href : null)
                     .filter(link => link !== null);
    });
    console.log(`Nombre de personnages trouvés : ${characterLinks.length}`);

    const results = [];

    // Itération sur chaque lien de personnage
    for (let i = 0; i < characterLinks.length; i++) {
      const link = characterLinks[i];
      console.log(`Traitement du personnage ${i + 1} : ${link}`);

      // Extraction du nom (dernier segment de l'URL)
      const urlName = link.split('/').pop();
      // Nettoyage du nom pour enlever les caractères indésirables (par exemple "%")
      const name = urlName.replace(/[^a-zA-Z0-9-_]/g, '');
      // Vérification : si le dossier du personnage existe déjà, on passe ce personnage
      const dirPath = path.join('images', name);
      if (fs.existsSync(dirPath)) {
        console.log(`Le personnage ${name} a déjà été traité. Passage au suivant.`);
        continue;
      }

      // Navigation vers la page de détail du personnage
      await page.goto(link, { waitUntil: 'networkidle2' });

      let imageAvailable = true;
      // Vérifie s'il y a une image (timeout augmenté à 30000ms)
      try {
        await page.waitForSelector('img.portrait-225x350', { timeout: 30000 });
      } catch (imgError) {
        console.log(`Image non disponible pour ${name}.`);
        imageAvailable = false;
      }

      // Création du dossier pour ce personnage dans "images"
      fs.mkdirSync(dirPath, { recursive: true });
      console.log(`Répertoire créé : ${dirPath}`);
      // Note : plus de sous-dossier 'ref'. Les images seront stockées directement dans le répertoire du personnage.

      let imageUrl = '';
      // Si l'image est disponible, extraire son URL
      if (imageAvailable) {
        const details = await page.evaluate(() => {
          const imgElement = document.querySelector('img.portrait-225x350');
          const imageUrl = imgElement 
                           ? (imgElement.getAttribute('data-src') || imgElement.getAttribute('src') || '')
                           : '';
          return { imageUrl };
        });
        imageUrl = details.imageUrl;
        console.log("Nom extrait :", name);
        console.log("URL de l'image :", imageUrl);

        if (imageUrl) {
          // Détermination du nom de fichier à partir de l'URL de l'image (suppression de potentiels paramètres)
          const fileName = path.basename(imageUrl.split('?')[0]);
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

      // Retour à la page principale pour continuer
      await page.goto(url, { waitUntil: 'networkidle2' });
      await page.waitForSelector('h3.h3_character_name', { timeout: 10000 });
    }

    console.log("Tous les personnages extraits :", results);
  } catch (error) {
    console.error("Erreur lors du scraping :", error);
  } finally {
    await browser.close();
  }
})();