Saving and Loading tf.Model
This tutorial describes how to save and load models in TensorFlow.js. Saving and loading of models is an important capability. For example, how do you save the weights of a model fine-tuned by data only available in the browser (e.g., images and audio data from attached sensors), so that the model will be in a already-tuned state when the user loads the page again? Also consider the fact that the Layers API allows you to create models called tf.Models from scratch in the browser. How do you save the models created this way? These questions are addressed by the save/load API, available in TensorFlow.js since version 0.11.1.

NOTE: This document is about saving and loading tf.Models (i.e., Keras-style models in the tfjs-layers API). Saving and loading tf.FrozenModels (i.e., models loaded from TensoFlow SavedModels) are not supported yet and is being actively worked on.

Saving tf.Model
Let's begin with the most basic, hassle-free way of saving a tf.Model: to the Local Storage of the web browser. Local Storage is a standard client-side data store. Data saved there can persist across multiple loads of the same page. You can learn more about it at this MDN page.

Suppose you have a tf.Model object called model. Be it created by using the Layers API from scratch or loaded/fine-tuned from a pretrained Keras model, you can save it to Local Storage with one line of code:

const saveResult = await model.save('localstorage://my-model-1');
A few things are worth pointing out:

The save method takes a URL-like string argument that starts with a scheme. In this case we use the localstorage:// scheme to specify that the model is to be saved to Local Storage.
The scheme is followed by a path. In the case of saving to Local Storage, the path is just an arbitrary string that uniquely identifies the model being saved. It will be used, for example, when you load model back from Local Storage.
The save method is asynchronous, so you need to use then or await if its completion forms the precondition of other actions.
The return value of model.save is a JSON object that carries some potentially useful pieces of information, such as the byte sizes of the model's topology and weights.
Any tf.Model, regardless of whether it is constructed with tf.sequential and what types of layers it consists of, can be saved this way.
The table below lists all currently supported destinations of saving models an their respecitve schemes and examples.

Saving Destination	Scheme string	Code example
Local Storage (Browser)	localstorage://	await model.save('localstorage://my-model-1');
IndexedDB (Browser)	indexeddb://	await model.save('indexeddb://my-model-1');
Trigger file downloads (Browser)	downloads://	await model.save('downloads://my-model-1');
HTTP request (Browser)	http:// or https://	await model.save('http://model-server.domain/upload');
File system (Node.js)	file://	await model.save('file:///tmp/my-model-1');
We will expand on some of the saving routes in the following sections.

IndexedDB
IndexedDB is another client-side data store supported by most mainstream web browsers. Unlike Local Storage, it has better support for storing large binary data (BLOBs) and a greater size quota. Hence, saving tf.Models to IndexedDB will typically give you better storage efficiency and a larger size limit compared to Local Storage.

File Downloads
The string that follows the downloads:// scheme is a prefix for the names of files that will be downloaded. For example, the line model.save('downloads://my-model-1') will cause the browser to download two files sharing the same filename prefix:

A text JSON file named my-model-1.json, which carries the topology of the model in its modelTopology field and a manifest of weights in its weightsManifest field.
A binary file carrying the weight values named my-model-1.weights.bin.
These files are in the same format as the artifacts converted from Keras HDF5 files by tensorflowjs converter.

Note: some browsers require users to grant permissions before more than one file can be downloaded at the same time.

HTTP Request
If tf.Model.save is called with an HTTP/HTTPS URL, the topology and weights of the model will be sent to the specified HTTP server via a POST request. The body of the POST request has a format called multipart/form-data. It is a standard MIME format for uploading files to servers. The body consist of two files, with filenames model.json and model.weights.bin. The formats of the files are identical to those of the downloaded files triggered by the downloads:// scheme (see section above). This doc string contains a Python code snippet that demonstrates how one may use the flask web framework, together with Keras and TensorFlow, to handle the request originated from save and reconstitute the request's payload as a Keras Model object in the server's memory.

Often, your HTTP server has special constraints and requirements on requests, such as HTTP methods, headers and credentials for authentication. You can gain fine-grained control over these aspects of the requests from save by replacing the URL string argument with the calls to tf.io.browserHTTPRequest. It is a more verbose API, but it affords greater flexiblity in controlling HTTP requests originated by save. For example:

await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
Native File System
TensorFlow.js can be used from Node.js. See the tfjs-node project for more details. Unlike web browsers, Node.js can access the local file system directly. Therefore, you can save tf.Models to the file system, in pretty much the same way as how you save a model to disk in Keras. To do this, first make sure you have imported the @tensorflow/tfjs-node package, e.g., using Node.js's require syntax:

require('@tensorflow/tfjs-node');
After the importing, the file:// URL scheme can be used for model saving and loading. For model saving, the scheme is followed by the path to the directory in which the model artifacts are to be saved, for example:

await model.save('file:///tmp/my-model-1');
The command above will generate a model.json file and a weights.bin file in the /tmp/my-model-1 directory. These two files have the same format as the files described in the File Downloads and HTTP Request sections above. After the model is saved, it can be loaded back into a Node.js program running TensorFlow.js or served for the browser version of TensorFlow.js. To achieve the former, you call tf.loadModel() with the path to the model.json file:

const model = await tf.loadModel('file:///tmp/my-model-1/model.json');
To achieve the latter, serve the saved files as static files from a web server.

Loading tf.Model
The ability to save tf.Models will not be useful if the models can't be loaded back afterwards. Model loading is done by calling tf.loadModel, with a scheme-based URL-like string argument. The string argument is symmetrical to tf.Model.save in most cases. The table below gives a summary of the supported loading routes:

Loading Route	Scheme string	Example
Local Storage (Browser)	localstorage://	await tf.loadModel('localstorage://my-model-1');
IndexedDB (Browser)	indexeddb://	await tf.loadModel('indexeddb://my-model-1');
User-uploaded files (Browser)	N/A	await tf.loadModel(tf.io.browserFiles([modelJSONFile, weightsFile]));
HTTP request (Browser)	http:// or https://	await tf.loadModel('http://model-server.domain/download/model.json');
File system (Node.js)	file://	await tf.loadModel('file:///tmp/my-model-1/model.json');
In all the loading routes, tf.loadModel returns a (Promise of) a tf.Model object if the loading succeeds, and throw an Error if it fails.

Loading from Local Storage or IndexedDB is exactly symmetrical with respect to saving. However, loading from user-uploaded files is not perfectly symmetrical with respect to downloading files from the browser. In particular, the files uploaded by the user are not represented as URL-like strings. Instead, they are specified as an Array of File objects. A typical workflow is letting users select files from their local filesystem by using HTML file input elements such as

<input name="json-upload" type="file" />
<input name="weights-upload" type="file" />
These will appear as two "Choose file" buttons in the browser that users can use to select files. Once users have selected a model.json file and a weights file in the two file inputs respectively, the file objects will be available under the corresponding HTML elements, and they can be used to load a tf.Model as follows:

const jsonUpload = document.getElementById('json-upload');
const weightsUpload = document.getElementById('weights-upload');

const model = await tf.loadModel(
    tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
Loading a model from HTTP request is also slightly asymmetric with respect to saving a mode via HTTP request. In particular, tf.loadModel takes the URL or path to a model.json file, as shown in the example in the table above. This is an API that has existed since the initial release of TensorFlow.js.

Managing models stored in browser Local Storage and IndexedDB
As you have learned above, you can store a tf.Model's topology and weights in the user's client-side browser data stores, including Local Storage and IndexedDB, by using code such as model.save('localstorage://my-model') and model.save('indexeddb://my-model'). But how do you find out what models have been stored there so far? This can be achieved by using the model management methods that come with the tf.io API:

// List models in Local Storage.
console.log(await tf.io.listModels());
The return values of the listModels methods include not only the paths of the stored models, but also some brief meta-data about them, such as the byte sizes of their topology and weights.

The management API also enables you to copy, move or remove existing models. For example:

// Copy model from existing path to a new path.
// Copying between Local Storage and IndexedDB is supported.
tf.io.copyModel('localstorage://my-model', 'indexeddb://cloned-model');

// Move model from a path to another.
// Moving between Local Storage and IndexedDB is supported.
tf.io.moveModel('localstorage://my-model', 'indexeddb://cloned-model');

// Remove model.
tf.io.removeModel('indexeddb://cloned-model');
Converting saved tf.Models into Keras format
As described above, there are two approaches that allow you to save a tf.Model as files:

through file downloading from the web browser, using the downloads:// scheme
writing the model directly to the native file system in Node.js, using the file:// scheme. With tensorflowjs converter, you can convert these files into the HDF5 format that can then be loaded into Keras in Python. For example:
# Suppose you have downloaded `my-model-1.json`, accompanied by a weights file.

pip install tensorflowjs

tensorflowjs_converter \
    --input_format tensorflowjs --output_format keras \
    ./my-model-1.json /tmp/my-model-1.h5
