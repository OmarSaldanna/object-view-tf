// funcion para guardar el dataset del modelo knn
function saveModel() {
  // obtenemos el dataset del modelo y lo pasamos a string
  let str = JSON.stringify( Object.entries(knn.getClassifierDataset()).map(([label, data])=>[label, Array.from(data.dataSync()), data.shape]) );
  // lo guardamos en el localstorage
  localStorage.setItem("dataset_knn", str);
  // y mandamos una alerta
  M.toast({html: 'Modelo Guardado', classes: 'rounded teal white-text'});
}

function loadModel() {
  // obtenemos el dataset del localstorage
  let str = localStorage.getItem("dataset_knn");
  // lo convertimos a un formato json
  knn.setClassifierDataset( Object.fromEntries( JSON.parse(str).map(([label, data, shape])=>[label, tf.tensor(data, shape)]) ) );
  // mandamos una alerta
  M.toast({html: 'Datos Cargados', classes: 'rounded teal lighten-2 white-text'});
  // y ejecutamos la funcion principal
  main();
}
