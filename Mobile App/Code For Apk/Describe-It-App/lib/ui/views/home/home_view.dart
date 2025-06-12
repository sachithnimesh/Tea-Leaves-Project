import 'package:ccl_ui/ccl_ui.dart';
import 'package:flutter/material.dart';
import 'package:stacked/stacked.dart';

import 'home_viewmodel.dart';

class HomeView extends StackedView<HomeViewModel> {
  const HomeView({Key? key}) : super(key: key);

  @override
  Widget builder(BuildContext context, HomeViewModel viewModel, Widget? child) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: const Text('Tea Health Scanner',
            style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.w500)),
        centerTitle: true,
        backgroundColor: Colors.teal,
      ),
      body: BackgroundProgress<HomeViewModel>(
        child: Center(
          child: SingleChildScrollView(
            physics: BouncingScrollPhysics(),
            child: Column(
              spacing: sizeDefault,
              children: [
                verticalSpaceDefault,
                ElevatedButton(
                  onPressed: () => viewModel.onPressedPickFile(),
                  style: ButtonStyle(
                    backgroundColor:
                        const WidgetStatePropertyAll(Colors.black26),
                    minimumSize: const WidgetStatePropertyAll(Size(150, 50)),
                    shape: WidgetStatePropertyAll(RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10))),
                  ),
                  child: const Text("CHOOSE FILE",
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w500)),
                ),
                verticalSpaceDefault,
                Container(
                  height: MediaQuery.sizeOf(context).height * 0.1,
                  width: MediaQuery.sizeOf(context).width * 0.8,
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      border:
                          Border.all(width: 3, color: Colors.grey.shade400)),
                  child: viewModel.pickedFile != null
                      ? Center(
                          child: Text(
                            viewModel.fileName ?? "-",
                            textAlign: TextAlign.center,
                          ),
                        )
                      : const Center(
                          child: Text(
                            "No File Selected",
                            textAlign: TextAlign.center,
                            style: TextStyle(
                                color: Colors.black54,
                                fontSize: sizeDefault,
                                fontWeight: FontWeight.w300),
                          ),
                        ),
                ),
                verticalSpaceDefault,
                if (viewModel.tfLiteFile == true) ...[
                  ElevatedButton(
                    onPressed: () => viewModel.onPressedPickImage(true),
                    style: ButtonStyle(
                      backgroundColor:
                          const WidgetStatePropertyAll(Colors.black26),
                      minimumSize: const WidgetStatePropertyAll(Size(250, 50)),
                      shape: WidgetStatePropertyAll(RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10))),
                    ),
                    child: const Text("OPEN GALLERY",
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w500)),
                  ),
                  ElevatedButton(
                    onPressed: () => viewModel.onPressedPickImage(false),
                    style: ButtonStyle(
                      backgroundColor:
                          const WidgetStatePropertyAll(Colors.black26),
                      minimumSize: const WidgetStatePropertyAll(Size(250, 50)),
                      shape: WidgetStatePropertyAll(RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10))),
                    ),
                    child: const Text("START CAMERA",
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w500)),
                  ),
                  verticalSpaceSmall,
                  Container(
                    height: MediaQuery.sizeOf(context).height * 0.35,
                    width: MediaQuery.sizeOf(context).width * 0.8,
                    decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                            style: BorderStyle.solid,
                            width: 5,
                            color: Colors.grey.shade400)),
                    child: viewModel.pickedImage != null
                        ? Image.file(viewModel.pickedImage!)
                        : Image.asset("assets/images/ic_upload_image.png"),
                  ),
                  verticalSpaceSmall,
                  ElevatedButton(
                    onPressed: viewModel.onPressedDetect,
                    style: ButtonStyle(
                      backgroundColor:
                          const WidgetStatePropertyAll(Colors.teal),
                      minimumSize: const WidgetStatePropertyAll(Size(250, 50)),
                      shape: WidgetStatePropertyAll(RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10))),
                    ),
                    child: const Text("DETECT",
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w500)),
                  ),
                  verticalSpaceMedium,
                  ListView.separated(
                    separatorBuilder: (context, index) => verticalSpaceLight,
                    shrinkWrap: true,
                    itemCount: viewModel.labeledOutput.length,
                    padding: EdgeInsets.symmetric(horizontal: sizeDefault),
                    itemBuilder: (BuildContext context, int index) {
                      return Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            flex: 2,
                            child: Tooltip(
                              message:
                                  viewModel.labeledOutput.keys.elementAt(index),
                              child: Text(
                                viewModel.labeledOutput.keys.elementAt(index),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                    color: Colors.black54,
                                    fontSize: 18,
                                    fontWeight: FontWeight.w500),
                              ),
                            ),
                          ),
                          Expanded(
                            child: Tooltip(
                              message: viewModel.labeledOutput.values
                                  .elementAt(index)
                                  .toString(),
                              child: Text(
                                viewModel.labeledOutput.values
                                    .elementAt(index)
                                    .toString(),
                                style: const TextStyle(
                                    color: Colors.teal,
                                    fontSize: sizeDefault,
                                    fontWeight: FontWeight.w500),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                  verticalSpaceMedium
                ]
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  HomeViewModel viewModelBuilder(BuildContext context) => HomeViewModel();
}
