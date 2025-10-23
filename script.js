/* script.js — handles predictions loading & visualization */
div.className = 'pred-item';


const left = document.createElement('div');
left.className = 'pred-left';
left.innerHTML = `<div class='pulse-dot' aria-hidden='true'></div><div><div style='font-weight:700'>${item.name}</div><div class='muted'>${item.code}</div></div>`;


const right = document.createElement('div');
right.innerHTML = `<div class='percent'>${(item.prob*100).toFixed(1)}%</div><div class='muted' style='font-size:12px'>${item.points} pts</div>`;


div.appendChild(left);div.appendChild(right);
predList.appendChild(div);
});


// Chart
const labels = items.map(i=>i.name);
const values = items.map(i=>Math.round(i.prob*1000)/10);


const ctx = document.getElementById('probChart').getContext('2d');
if(chart) chart.destroy();
chart = new Chart(ctx, {
type: 'bar',
data: {
labels: labels,
datasets: [{
label: 'Champion Probability (%)',
data: values,
backgroundColor: labels.map((l,i)=>`rgba(255,46,46,${0.9 - i*0.08})`),
borderRadius:6
}]
},
options: {
animation: { duration: 800 },
plugins: { legend: { display:false } },
scales: {
y: { beginAtZero:true, ticks:{color:'#cfd8e3'} },
x: { ticks:{color:'#cfd8e3'} }
}
}
});
}


async function loadPredictions(source='sample'){
try{
if(source === 'sample'){
// small delay for UX
await new Promise(r=>setTimeout(r,350));
renderPredictions(sampleData);
} else {
// fetch predictions.json saved in repo root
const res = await fetch(source,{cache:'no-store'});
if(!res.ok) throw new Error('Failed to load predictions.json — place file in site root');
const json = await res.json();
// expect format: { 'PIA': { name:'Piastri', points:346, prob:0.72 }, ... }
renderPredictions(json);
}
}catch(err){
console.error(err);
// show sample on error
renderPredictions(sampleData);
}
}


// buttons
document.getElementById('run-sim').addEventListener('click', ()=>loadPredictions(document.getElementById('sourceSelect').value));
document.getElementById('refreshBtn').addEventListener('click', ()=>loadPredictions(document.getElementById('sourceSelect').value));


// initial
window.addEventListener('load', ()=>{
ScrollReveal().reveal('.hero-inner',{delay:120,distance:'20px',origin:'bottom'});
ScrollReveal().reveal('.panel',{delay:160,distance:'16px',origin:'bottom'});
loadPredictions('sample');
});