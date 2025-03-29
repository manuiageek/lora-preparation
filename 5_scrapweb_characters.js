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
  // URL de la page listant les personnages de Cat's Eye
  const url = 'https://myanimelist.net/anime/2043/Cats_Eye/characters';

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

      // Navigation vers la page de détail du personnage
      await page.goto(link, { waitUntil: 'networkidle2' });
      // Attente de l'image en haute résolution
      await page.waitForSelector('img.portrait-225x350', { timeout: 10000 });

      // Extraction du nom (dernier segment de l'URL)
      const urlName = link.split('/').pop();
      // Extraction de l'URL de l'image depuis la page du personnage
      const details = await page.evaluate(() => {
        const imgElement = document.querySelector('img.portrait-225x350');
        const imageUrl = imgElement 
                         ? (imgElement.getAttribute('data-src') || imgElement.getAttribute('src') || '')
                         : '';
        return { imageUrl };
      });

      const name = urlName;
      console.log("Nom extrait :", name);
      console.log("URL de l'image :", details.imageUrl);

      // Création du dossier pour ce personnage dans "images"
      const dirPath = path.join('images', name);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`Répertoire créé : ${dirPath}`);
      }
      // Création du sous-dossier "ref"
      const refDir = path.join(dirPath, 'ref');
      if (!fs.existsSync(refDir)) {
        fs.mkdirSync(refDir, { recursive: true });
        console.log(`Sous-dossier 'ref' créé : ${refDir}`);
      }

      // Détermination du nom de fichier à partir de l'URL de l'image (supprime d'éventuels paramètres)
      const fileName = path.basename(details.imageUrl.split('?')[0]);
      const filePath = path.join(refDir, fileName);

      try {
        console.log(`Téléchargement de l'image vers ${filePath}`);
        await downloadImage(details.imageUrl, filePath);
        console.log("Téléchargement réussi !");
      } catch (downloadError) {
        console.error("Erreur lors du téléchargement de l'image :", downloadError);
      }

      results.push({ url: link, name, imageUrl: details.imageUrl });

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