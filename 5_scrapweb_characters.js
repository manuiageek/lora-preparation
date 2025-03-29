import puppeteer from 'puppeteer';

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
    // Charge la page et attend les éléments contenant les noms de personnages
    await page.goto(url, { waitUntil: 'networkidle2' });
    await page.waitForSelector('h3.h3_character_name', { timeout: 10000 });
    console.log("La page des personnages est chargée, début de l'extraction des liens...");

    // Extraction de tous les liens des pages de détails des personnages
    const characterLinks = await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('h3.h3_character_name'));
      // On récupère le href du parent (la balise <a>)
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
      // Attendre que l'image en haute résolution soit chargée
      await page.waitForSelector('img.portrait-225x350', { timeout: 10000 });

      // Extrait le nom depuis l'URL : le dernier segment de l'URL (ici "Ai_Kisugi")
      const urlName = link.split('/').pop();

      // Extraction facultative des autres détails depuis la page du personnage.
      const details = await page.evaluate(() => {
        const imgElement = document.querySelector('img.portrait-225x350');
        const imageUrl = imgElement 
                         ? (imgElement.getAttribute('data-src') || imgElement.getAttribute('src') || '')
                         : '';
        return { imageUrl };
      });

      // Ici, on remplace "name" par le segment de l'URL.
      const name = urlName;
      console.log("Nom extrait :", name);
      console.log("Détails extraits :", details);
      results.push({ url: link, name, ...details });

      // Retour à la page principale pour le personnage suivant
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