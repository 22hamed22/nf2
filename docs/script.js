// Minimal JS enhancements for docs/index.html
document.addEventListener("DOMContentLoaded", function () {
  // Open external links in a new tab
  document.querySelectorAll("a").forEach((a) => {
    const href = a.getAttribute("href");
    if (!href) return;
    // If link points outside the docs folder, open in a new tab
    if (href.startsWith("http") || href.startsWith("https") || href.startsWith("../notebooks/")) {
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    }
  });

  // Simple keyboard shortcut: press "g" to go to the refactored notebook link if present
  document.addEventListener("keydown", function (e) {
    if (e.key === "g" || e.key === "G") {
      const link = document.querySelector('a[href$="full_nerf_example_refactored.ipynb"]');
      if (link) window.open(link.href, "_blank");
    }
  });
});