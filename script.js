const model = tf.sequential();

model.add(
    tf.layers.dense({
        units: 1,
        inputShape: [9]
    })
);

model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd',
    metrics: ['accuracy']
});


const xs = tf.tensor2d([
    [0, 1, 0,
     1, 1, 1,
     0, 1, 0],
    [0, 0, 0,
     1, 1, 1,
     0, 0, 0],
    [1, 0, 1,
     0, 1, 0,
     1, 0, 1],
    [0, 0, 1,
     0, 1, 0,
     1, 0, 0]
], [4, 9]);
const ys = tf.tensor2d([1, 2, 3, 4], [4, 1]);

async function entrenar() {
    const carga = document.getElementById('carga');
    const mensaje = document.getElementById('mensaje');
    const bar = document.getElementById('bar');
    
    await model.fit(xs, ys, {
        epochs: 500,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                bar.style.transform = `translateX(${epoch / 5 - 100}%)`;
                mensaje.innerHTML = `Entrenando IA <span>${Math.round(epoch / 5)}%</span>`;
            }
        }
    });
    await new Promise(resolve => {
        setTimeout(() => {
            carga.style.opacity = 0;
            resolve();
        }, 0);
    });
    await new Promise(resolve => {
        setTimeout(() => {
            carga.style.display = 'none';
            resolve();
        }, 1000);
    });
}

entrenar();

function predecir() {
    const dato = parseInt(document.getElementById('dato').value);
    const mensaje = document.getElementById('mensaje');
    const output = model.predict(tf.tensor2d([dato], [1, 1]));
    console.log(Math.floor(output.dataSync()[0]));
    switch (output.dataSync()[0]) {
        case 1:
            mensaje.innerHTML = 'Es una suma';
            break;
        case 2:
            mensaje.innerHTML = 'Es una resta';
            break;
        case 3:
            mensaje.innerHTML = 'Es una multiplicación';
            break;
        case 4:
            mensaje.innerHTML = 'Es una división';
            break;
    }
}