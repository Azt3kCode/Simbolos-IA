model = tf.sequential();

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

entrenar();

async function entrenar() {
    const carga = document.getElementById('carga');
    const mensaje = document.getElementById('mensaje');
    const bar = document.getElementById('bar');
    
    await model.fit(xs, ys, {
        epochs: 1000,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                bar.style.transform = `translateX(${epoch / 10 - 100}%)`;
                mensaje.innerHTML = `Entrenando IA <span>${Math.round(epoch / 10)}%</span>`;
                /*console.log(epoch, 
                    model.predict(tf.tensor2d([
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
                    ], [4, 9])).dataSync());*/
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

let array = [0, 0, 0, 0, 0, 0, 0, 0, 0];

async function predecir() {
    const mensaje = document.getElementById('msg');
    const output = await model.predict(tf.tensor2d(array, [1, 9]));
    // redondear output al numero enteros mas cercano
    switch (Math.round(await output.dataSync()[0])) {
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
        default:
            mensaje.innerHTML = 'No se pudo predecir';
            break;
    }
    console.log(mensaje);
}

async function cambiar(id) {
    const el = document.getElementById(id);
    if (array[id - 1] === 0) {
        el.style.background = '#0099ff';
        el.style.borderBottom = '5px solid #0088ee';
        el.style.transform = 'scale(1.1)';
        array [id - 1] = 1;
    } else {
        el.style.background = '#eee';
        el.style.borderBottom = '5px solid #ddd';
        el.style.transform = 'scale(1)';
        array [id - 1] = 0;
    }
    await predecir();
}