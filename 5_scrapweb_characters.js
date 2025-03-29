import puppeteer from 'puppeteer';

(async () => {
  // URL à scraper
  const url = 'https://myanimelist.net/anime/49785/Fairy_Tail__100-nen_Quest/characters';

  // Lance le navigateur (mode headless)
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Va sur la page et attend que le réseau soit inactif (pour s'assurer du chargement complet)
  await page.goto(url, { waitUntil: 'networkidle2' });

  // Exécute du code dans le contexte de la page pour extraire les noms des personnages
  const characters = await page.evaluate(() => {
    // Récupère tous les liens qui contiennent la classe "hoverinfo_trigger"
    const anchors = Array.from(document.querySelectorAll('a.hoverinfo_trigger'));
    // Retourne le texte nettoyé de chacun de ces liens
    return anchors.map(anchor => anchor.textContent.trim()).filter(name => name !== '');
  });

  // Affiche les noms des personnages dans la console
  console.log("Noms des personnages :", characters);

  // Ferme le navigateur
  await browser.close();
})();
