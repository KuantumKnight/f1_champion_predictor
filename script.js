// Example static prediction (replace later with live data)
const predictions = {
  "VER": 0.58,
  "LEC": 0.22,
  "NOR": 0.15,
  "HAM": 0.05
};

function displayPredictions() {
  const container = document.getElementById("predictions");
  container.innerHTML = "";

  for (const [driver, prob] of Object.entries(predictions)) {
    const bar = document.createElement("div");
    bar.classList.add("bar");
    bar.style.width = (prob * 100) + "%";
    bar.textContent = `${driver} – ${(prob * 100).toFixed(1)}%`;
    container.appendChild(bar);
  }
}

document.addEventListener("DOMContentLoaded", displayPredictions);
