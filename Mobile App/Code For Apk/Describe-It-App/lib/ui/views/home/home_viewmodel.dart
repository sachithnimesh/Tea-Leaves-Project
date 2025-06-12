import 'dart:io';
import 'package:describe_it_app/app/app.locator.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:stacked/stacked.dart';
import 'package:stacked_services/stacked_services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class HomeViewModel extends BaseViewModel {
  Interpreter? interpreter;
  File? pickedImage;
  File? pickedFile;
  final ImagePicker _picker = ImagePicker();
  String? fileName;
  String? _result;
  String? get result => _result;
  bool? tfLiteFile;

  List<String> labels = [
    'Healthy',
    'Tea leaf blight',
    'Tea red leaf spot',
    'Tea red scab'
  ];

  Map<String, double> labeledOutput = {};

  SnackbarService _snackbarService = locator.get();

  // Pick an image from gallery or camera
  Future<void> onPressedPickImage(bool isGallery) async {
    final XFile? pickedFile = await _picker.pickImage(
        source: isGallery ? ImageSource.gallery : ImageSource.camera);
    if (pickedFile != null) {
      pickedImage = File(pickedFile.path);
      notifyListeners();
    }
  }

  Future<void> onPressedPickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowMultiple: false,
      type: FileType.any,
    );
    if (result != null) {
      pickedFile = File(result.files.first.path!);
      fileName = result.files.first.name;
      loadModel();
      if (result.files.first.extension == "tflite") {
        tfLiteFile = true;
      } else {
        tfLiteFile = false;
        _snackbarService.showSnackbar(
          message: "please select a tflite file",
          duration: const Duration(seconds: 3),
        );
      }
      notifyListeners();
    }
  }

  // Load the TFLite model
  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromFile(pickedFile!);
    } catch (e) {
      _snackbarService.showSnackbar(
          message: "$e", duration: const Duration(seconds: 4));
    }
  }

  // Preprocess the image and run inference
  void runInference() {
    try {
      setBusy(true);
      if (interpreter == null || pickedImage == null) {
        _snackbarService.showSnackbar(
          message: "please select an image",
          duration: const Duration(seconds: 1),
        );
        return;
      }

      // Load and preprocess the image
      img.Image? image = img.decodeImage(pickedImage!.readAsBytesSync());
      if (image == null) return;

      // Resize image
      img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

      // Convert to tensor format (example: normalize to [0, 1])
      var input = List.generate(
        1,
        (_) => List.generate(
          224,
          (i) => List.generate(
            224,
            (j) => [
              resizedImage.getPixel(j, i).r / 255.0,
              resizedImage.getPixel(j, i).g / 255.0,
              resizedImage.getPixel(j, i).b / 255.0,
            ],
          ),
        ),
      );

      var output = List.filled(1 * 4, 0.0).reshape([1, 4]);

      interpreter!.run(input, output);

      for (int i = 0; i < labels.length; i++) {
        double x = output[0][i];
        labeledOutput[labels[i]] = double.parse(x.toStringAsFixed(10));
      }

      List<MapEntry<String, double>> sortedEntriesDesc =
          labeledOutput.entries.toList();
      sortedEntriesDesc.sort((a, b) => b.value.compareTo(a.value));

      labeledOutput = Map.fromEntries(sortedEntriesDesc);

      String result = processOutput(labeledOutput);
      _result = result;
      notifyListeners();
    } catch (e) {
      _snackbarService.showSnackbar(
        message: "$e",
        duration: const Duration(seconds: 4),
      );
    } finally {
      setBusy(false);
    }
  }

  // Process the model output to text
  String processOutput(Map<String, double> output) {
    return 'Extracted text: $output';
  }

  void onPressedDetect() {
    runInference();
  }
}
