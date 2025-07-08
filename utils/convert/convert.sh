echo $PATH
source utils/convert/venv/bin/activate
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --signature_name=serving_default \
  --saved_model_tags=serve \
  public/model/saved_model \
  public/model/tfjs/
python -m tf2onnx.convert \
  --saved-model public/model/saved_model \
  --output public/model/mnist.onnx \
  --opset 13
