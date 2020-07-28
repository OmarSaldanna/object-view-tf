// definimos en global el modelo clasificador: mobilenet
let net;
// y tambien el knn
let knn = knnClassifier.create();

// numero de gestos o labels, son 6 incluyendo el no entrenado
number_labels = 6;

// ahora los elementos: la webcam
const webcamEl = document.getElementById('webcam');


// funcion para cambiar el detectando
function changeView(index) {
  // usamos el try por que asi solo da error
  try {
    // iteramos los items de la coleccion
    for (i = 0; i <= number_labels; i++) {
      // si damos con el index, lo activamos
      if (i == index) {
        document.getElementById('view_' + i).classList.add('active');
      }
      // y a todos los demas los desactivamos
      else {
        document.getElementById('view_' + i).classList.remove('active');
      }
    }
  } catch (error) { }
}

// funcion para agregar ejemplos al knn
async function addExample(classId) {
  // capturamos la webcam
  const img = await webcam.capture();
  // hacemos la inferencia que retorna un tensor numerico
  const activation = net.infer(img, true);
  // agragamos el ejemplo y su label que es un numero
  knn.addExample(activation, classId);
  // y destruimos el objeto de memoria
  img.dispose();
}

// funcion principal
async function main() {
  // traemos el modelo de mobilenet
  net = await mobilenet.load();

  // activamos la webcam
  webcam = await tf.data.webcam(webcamEl);

  // ciclo infinito para detectar con la webcam
  while (true) {
    // capturamos la webcam
    const img = await webcam.capture();

    // esta regresara un tensor numerico como resultado
    const activation_knn = net.infer(img, "conv_preds");

    // classificacion por knn, para detectar gestos locales
    try {
      // clasificamos el tensor que llego
      result_knn = await knn.predictClass(activation_knn);
      
      // y cambiamos al label correspondiente
      changeView(result_knn.label);

    } catch (error) { }

    // destruimos el objeto de la memoria para limpiar
    img.dispose();

    // esperamos a procesar el frame, para no sobreprocesar
    await tf.nextFrame();
  }
}

// y ejecutamos la funcion principal
main()