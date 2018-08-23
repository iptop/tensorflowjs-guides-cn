Importing a TensorFlow GraphDef based Models into TensorFlow.js
TensorFlow GraphDef based models (typically created via the Python API) may be saved in one of following formats:

TensorFlow SavedModel
Frozen Model
Session Bundle
Tensorflow Hub module
All of above formats can be converted by TensorFlow.js converter to TensorFlow.js web friendly format, which can be loaded directly into TensorFlow.js for inference.

(Note: TensorFlow has deprecated session bundle format, please migrate your models to SavedModel format.)

Requirements
The conversion procedure requires a Python environment; you may want to keep an isolated one using pipenv or virtualenv. To install the converter, and run following:

pip install tensorflowjs
Importing a TensorFlow models into TensorFlow.js is a two-step process. First, convert an existing model to TensorFlow.js web format, and then load it into TensorFlow.js.

Step 1. Convert an existing TensorFlow model to TensorFlow.js Web format
Run the converter script provided by the pip package:

Usage: SavedModel example:

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
Frozen model example:

tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
Session bundle model example:

tensorflowjs_converter \
    --input_format=tf_session_bundle \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/session_bundle \
    /mobilenet/web_model
Tensorflow Hub module example:

tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
Positional Arguments	Description
input_path	Full path of the saved model directory, session bundle directory, frozen model file or TensorFlow Hub module handle or path.
output_path	Path for all output artifacts.
Options	Description
--input_format	The format of input model, use tf_saved_model for SavedModel, tf_frozen_model for frozen model, tf_session_bundle for session bundle, tf_hub for TensorFlow Hub module and keras for Keras HDF5.
--output_node_names	The names of the output nodes, separated by commas.
--saved_model_tags	Only applicable to SavedModel conversion, Tags of the MetaGraphDef to load, in comma separated format. Defaults to serve.
--signature_name	Only applicable to TensorFlow Hub module conversion, signature to load. Defaults to default. See https://www.tensorflow.org/hub/common_signatures/.
Use following command to get the detail help message:

tensorflowjs_converter --help
Converter generated files
The conversion script above produces 3 types of files:

web_model.pb (the dataflow graph)
weights_manifest.json (weight manifest file)
group1-shard\*of\* (collection of binary weight files)
For example, here is the MobileNet model converted and served in following location:

https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/optimized_model.pb
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/weights_manifest.json
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard1of5
...
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard5of5
Step 2: Loading and running in the browser
Install the tfjs-converter npm package
yarn add @tensorflow/tfjs or npm install @tensorflow/tfjs

Instantiate the FrozenModel class and run inference.
import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'https://.../mobilenet/web_model.pb';
const WEIGHTS_URL = 'https://.../mobilenet/weights_manifest.json';

const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
const cat = document.getElementById('cat');
model.execute({input: tf.fromPixels(cat)});
Check out our working MobileNet demo.

If your server requests credentials for accessing the model files, you can provide the optional RequestOption param, which will be directly passed to the fetch function call.

const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL,
    {credentials: 'include'});
Please see fetch() documentation for details.

Supported operations
Currently TensorFlow.js only supports a limited set of TensorFlow Ops. See the full list. If your model uses any unsupported ops, the tensorflowjs_converter script will fail and produce a list of the unsupported ops in your model. Please file issues to let us know what ops you need support for.

Loading the weights only
If you prefer to load the weights only, you can use follow code snippet.

import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
