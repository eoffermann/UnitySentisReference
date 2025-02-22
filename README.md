# Sentis 2.1.2 for Unity – Detailed Documentation

Sentis is a neural network inference library for Unity that enables you to load and execute models (primarily in ONNX format) at runtime on both the CPU and GPU. The following documentation provides an in‐depth look at Sentis 2.1.2, including its design philosophy, core classes, the functional API, and best practices.

*This is not official Unity documentation. It is based on my own work, and on inference from various language models processing code and documentation for Sentis 2.1.2*
---

## Table of Contents

1. [Introduction and Core Concepts](#introduction-and-core-concepts)
2. [Installation and Setup](#installation-and-setup)
3. [Model Loading and Runtime Model Creation](#model-loading-and-runtime-model-creation)
4. [Tensors and Tensor Operations](#tensors-and-tensor-operations)
5. [Worker and Inference Execution](#worker-and-inference-execution)
6. [The Functional API](#the-functional-api)
7. [Texture Conversion Utilities](#texture-conversion-utilities)
8. [Model Asset Import Settings](#model-asset-import-settings)
9. [Advanced Topics and Best Practices](#advanced-topics-and-best-practices)
10. [Sample Code Projects](#sample-code-projects)
11. [API Reference Highlights](#api-reference-highlights)
12. [Troubleshooting and Performance Considerations](#troubleshooting-and-performance-considerations)

---

## 1. Introduction and Core Concepts

Sentis is built to offer real-time neural network inference within Unity applications. Key concepts include:

- **Neural Network Inference:** Run trained ONNX models in real time.
- **ONNX Format Support:** Accepts models with opset versions 7 to 15.
- **Tensors:** Multi-dimensional arrays (up to eight dimensions) holding data as `float` or `int`.
- **Workers:** Inference engines that execute models on selected backends (CPU or GPU).
- **Backend Types:** Three primary backends are available:
  - **CPU:** Using Burst compilation.
  - **GPUCompute:** Utilizing compute shaders (the fastest option when supported).
  - **GPUPixel:** Using pixel shaders (fallback when compute shaders aren’t available).
- **Functional API:** A graph-based API that lets you modify or extend a model (for example, to add postprocessing like softmax).
- **Model Optimization:** Sentis can optimize the runtime model when using the functional API.

For an overview of these concepts, see the [Sentis overview](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/index.html) on Unity’s documentation.

---

## 2. Installation and Setup

Sentis is available via the Unity Package Manager. To install it:

1. Open Unity and go to **Window > Package Manager**.
2. Find the **Sentis** package (version 2.1.2) and click **Install**.
3. (Optional) Import sample projects by selecting the **Samples** tab and clicking **Import** for a given sample.

These samples provide useful guidance on everything from basic tensor conversion to advanced model execution. [Samples](https://docs.unity3d.com/Packages/com.unity.sentis%402.1/manual/package-samples.html)

---

## 3. Model Loading and Runtime Model Creation

### 3.1. Using ModelAsset and ModelLoader

Sentis expects your neural network models to be imported as **ModelAsset** objects (typically ONNX files). The `ModelLoader` class converts these assets into a runtime **Model**.

**Example – Basic Model Loading:**

```csharp
using UnityEngine;
using Unity.Sentis;

public class ModelLoaderExample : MonoBehaviour
{
    // Drag your ONNX model asset in the Inspector.
    public ModelAsset modelAsset;
    private Model runtimeModel;

    void Start()
    {
        // Load the runtime model from the asset.
        runtimeModel = ModelLoader.Load(modelAsset);
        Debug.Log("Model loaded successfully.");
    }

    void OnDestroy()
    {
        // Always clean up to prevent memory leaks.
        runtimeModel?.Dispose();
    }
}
```

In this example, the model asset is loaded at startup, and the returned runtime model is later disposed of when no longer needed. This basic pattern is common in Sentis applications. [Overview](https://docs.unity3d.com/Packages/com.unity.sentis%40latest/)

---

## 4. Tensors and Tensor Operations

Tensors are the primary data containers in Sentis. They represent multi-dimensional arrays and are used both as inputs and outputs for models.

### 4.1. Creating Tensors

You can create a tensor directly from an array or from other data sources (e.g., textures):

**Example – Creating a Tensor from an Array:**

```csharp
using Unity.Sentis;

public class TensorCreationExample
{
    public void CreateArrayTensor()
    {
        // Define the tensor shape (e.g., a vector with 4 elements).
        TensorShape shape = new TensorShape(4);
        int[] data = { 1, 2, 3, 4 };
        
        // Create a tensor with the given shape and data.
        Tensor<int> tensor = new Tensor<int>(shape, data);
        Debug.Log("Tensor created with shape: " + tensor.shape);
        
        // Remember to dispose when done.
        tensor.Dispose();
    }
}
```

**Example – Converting a Texture to a Tensor:**

```csharp
using UnityEngine;
using Unity.Sentis;

public class TextureTensorExample : MonoBehaviour
{
    public Texture2D inputTexture;

    void Start()
    {
        // Convert a Texture2D to a Tensor<float>.
        Tensor<float> tensor = TextureConverter.ToTensor(inputTexture);
        Debug.Log("Texture converted to tensor with shape: " + tensor.shape);
        
        // Dispose of the tensor after use.
        tensor.Dispose();
    }
}
```

### 4.2. Tensor Manipulation

Once a tensor is created, you can reshape it, index into it, or download its data:

- **Reshape:** Change the dimensions without modifying the underlying data.
  
  ```csharp
  tensor.Reshape(new TensorShape(2, 2));
  ```

- **Indexing:** When the tensor is on the CPU and writable, you can access elements directly:
  
  ```csharp
  tensor[0] = 5.2f;
  float value = tensor[0];
  ```

- **Downloading Data:** Retrieve data from the tensor into a native or managed array.
  
  ```csharp
  float[] dataArray = tensor.DownloadToArray();
  ```

For asynchronous scenarios (such as GPU output), use `ReadbackAndCloneAsync()` to avoid blocking the main thread. (https://docs.unity.cn/Packages/com.unity.sentis%402.1/api/Unity.Sentis.html)

---

## 5. Worker and Inference Execution

The **Worker** class encapsulates the inference engine. It handles sending the model’s operations to either the CPU or GPU and retrieving the results.

### 5.1. Creating and Using a Worker

A worker is created by passing in a runtime model and a specified backend type:

```csharp
using Unity.Sentis;

public class WorkerExample : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;

    void Start()
    {
        // Load the runtime model.
        runtimeModel = ModelLoader.Load(modelAsset);
        
        // Create a worker using the GPUCompute backend.
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        
        // Create your input tensor (this example uses an array).
        TensorShape shape = new TensorShape(1, 3, 224, 224);
        float[] inputData = new float[1 * 3 * 224 * 224]; // Populate this array with your data.
        Tensor<float> inputTensor = new Tensor<float>(shape, inputData);

        // Schedule the model for execution.
        worker.Schedule(inputTensor);
        
        // Retrieve output.
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        float[] results = outputTensor.DownloadToArray();
        Debug.Log("Inference complete. First output value: " + results[0]);
        
        // Dispose of tensors and worker when finished.
        inputTensor.Dispose();
        outputTensor.Dispose();
    }

    void OnDestroy()
    {
        worker?.Dispose();
        runtimeModel?.Dispose();
    }
}
```

In this example, the worker is used to run inference synchronously. For asynchronous output handling (especially on GPU), you can use methods like `ReadbackAndCloneAsync()` to avoid stalling the main thread.

### 5.2. Advanced Worker Methods

- **Multiple Inputs:**  
  ```csharp
  Tensor[] inputs = new Tensor[] { inputTensor1, inputTensor2 };
  worker.Schedule(inputs);
  ```
- **Copying Outputs:**  
  Use `CopyOutput` to transfer data into an externally owned tensor.
  
  ```csharp
  Tensor myOutputTensor;
  worker.CopyOutput("output", ref myOutputTensor);
  ```
- **Layer-By-Layer Execution:**  
  When debugging or performing custom operations, you can execute the model one layer at a time using `ScheduleIterable()`.  
  *(For details, consult the official samples.)*  

---

## 6. The Functional API

The Functional API in Sentis provides a way to modify or extend models by building a computational graph. This is useful for adding preprocessing, postprocessing, or even entirely new layers.

### 6.1. Building a Functional Graph

A typical workflow involves:
1. Creating a **FunctionalGraph**.
2. Adding model inputs via `AddInputs()`.
3. Applying model layers using `Functional.Forward()`.
4. Postprocessing (e.g., applying a softmax) using methods such as `Functional.Softmax()`.
5. Compiling the modified graph back into a runtime model.

**Example – Extending a Model with Softmax:**

```csharp
using UnityEngine;
using Unity.Sentis;

public class FunctionalAPIDemo : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;

    void Start()
    {
        // Load the original model.
        Model sourceModel = ModelLoader.Load(modelAsset);
        
        // Create a new functional graph.
        FunctionalGraph graph = new FunctionalGraph();
        
        // Add the model's inputs to the graph.
        FunctionalTensor[] inputs = graph.AddInputs(sourceModel);
        
        // Run the model's forward pass.
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs);
        
        // Apply softmax to the first output.
        FunctionalTensor softmaxOutput = Functional.Softmax(outputs[0]);
        
        // Compile the graph into a new model.
        runtimeModel = graph.Compile(softmaxOutput);
        
        // Create input data (for example, from a texture).
        Texture2D inputTexture = Resources.Load<Texture2D>("image-file");
        using Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture);
        
        // Create the worker with your preferred backend.
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        worker.Schedule(inputTensor);
        
        // Retrieve and process the output.
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        float[] predictions = outputTensor.DownloadToArray();
        Debug.Log("Prediction for class 0: " + predictions[0]);
    }

    void OnDestroy()
    {
        worker?.Dispose();
        runtimeModel?.Dispose();
    }
}
```

This example demonstrates the power of the Functional API to seamlessly integrate postprocessing directly into your model’s computation graph. [Workflow Example](https://docs.unity3d.com/Packages/com.unity.sentis%402.1/manual/workflow-example.html)

---

## 7. Texture Conversion Utilities

The **TextureConverter** class provides methods to convert between Unity textures and tensors. This is essential when your input data comes from camera feeds, images, or render textures.

### 7.1. Converting a Texture to a Tensor

```csharp
// Convert a Texture2D or RenderTexture to a Tensor<float>.
Tensor<float> tensor = TextureConverter.ToTensor(inputTexture);
// Optionally specify dimensions and channels:
Tensor<float> resizedTensor = TextureConverter.ToTensor(inputTexture, width: 28, height: 28, channels: 1);
```

### 7.2. Converting a Tensor Back to a Texture

```csharp
// Convert a tensor to a RenderTexture.
RenderTexture outputRT = TextureConverter.ToTexture(outputTensor);

// Render tensor data to an existing RenderTexture.
TextureConverter.RenderToTexture(outputTensor, outputRT);

// Render tensor output directly to the screen.
TextureConverter.RenderToScreen(outputTensor);
```

These utilities streamline the process of interfacing between machine learning data and Unity’s rendering system. 

---

## 8. Model Asset Import Settings

When you import an ONNX model as a **ModelAsset**, Unity provides an import settings window that details:

- **Inputs:**  
  - *name, index, shape, dataType*
- **Outputs:**  
  - *name, index, shape, dataType*
- **Layers:**  
  - *type, index, inputs* (defines execution order)
- **Constants:**  
  - *type, index, weights (tensor shape)*

Understanding these settings is key to debugging model compatibility issues and ensuring the input tensor’s shape matches model expectations.

---

## 9. Advanced Topics and Best Practices

### 9.1. Memory Management

- **Dispose Resources:** Always call `.Dispose()` on **Worker** and **Tensor** objects when finished to prevent memory leaks.
- **Asynchronous Execution:** Use `ReadbackAndCloneAsync()` or the readback request methods to ensure GPU computations do not block the main thread.

### 9.2. Performance Optimization

- **Profiling:** Utilize Unity’s Profiler window to track model performance.
- **Backend Selection:** Choose between CPU and GPU backends based on device capabilities. GPUCompute is typically the fastest, but compatibility should be verified.
- **Model Complexity:** Model performance will vary with the complexity of the network operators. Refer to the list of [Supported ONNX Operators](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/index.html) for guidelines. 

### 9.3. Tensor Formats and Data Layout

Different models may expect inputs in different tensor layouts (e.g., NHWC vs. NCHW). Use the Functional API or adjust your input tensor accordingly to match your model’s requirements.

---

## 10. Sample Code Projects

Unity provides a number of sample projects that demonstrate various Sentis use cases. These include:

- **Digit Recognition Sample:** Uses a handwritten digit recognition model.
- **Depth Estimation and Object Detection Samples:** Illustrate using the GPU for image processing and model inference.
- **Advanced Functional API Examples:** Show how to modify an existing model’s graph.

Refer to the [Sentis Samples repository](https://github.com/Unity-Technologies/sentis-samples) for full projects and detailed walkthroughs.

---

## 11. API Reference Highlights

Below is a quick rundown of key classes and methods:

### 11.1. ModelLoader

- **`ModelLoader.Load(ModelAsset modelAsset)`**  
  Converts a model asset to a runtime **Model**.

### 11.2. Worker

- **Constructor:**  
  `Worker(Model model, BackendType backendType)`  
  Creates a worker for a specified model and hardware backend.
- **Scheduling:**  
  `Schedule(Tensor inputTensor)` and `Schedule(Tensor[] inputs)`
- **Output Retrieval:**  
  `PeekOutput()`, `CopyOutput(string name, ref Tensor tensor)`
- **Resource Management:**  
  `Dispose()`
- **Iterative Execution:**  
  `ScheduleIterable()`

### 11.3. Tensor and TensorShape

- **Tensor Creation:**  
  `Tensor(TensorShape shape, float[] data)`  
  `Tensor(TensorShape shape, int[] data)`
- **Indexing, Reshaping, and Data Download:**  
  Use the provided methods for manipulating tensor data.

### 11.4. TextureConverter

- **ToTensor / ToTexture:**  
  Convert textures to tensors and vice versa.
- **Rendering:**  
  Methods to render tensors directly to screens or textures.

### 11.5. Functional API Classes

- **FunctionalGraph:**  
  Build and modify the computation graph.
- **Functional:**  
  Provides static methods like `Forward()`, `Softmax()`, `Constant()`, and others for graph manipulation.

For full API details, consult the [Sentis Scripting API documentation](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/api/index.html).

---

## 12. Troubleshooting and Performance Considerations

- **First-Run Overhead:**  
  The first call to functions like `TextureConverter.ToTensor()` may block due to shader compilation. Plan for this in your application startup.
- **Output Readback:**  
  When using GPU backends, ensure that you wait for asynchronous readbacks using `IsReadbackRequestDone()` or use `ReadbackAndCloneAsync()`.
- **Mismatch in Tensor Formats:**  
  Verify that the input tensor’s dimensions (e.g., NHWC vs. NCHW) match what the model expects.
- **Memory Leaks:**  
  Always dispose of workers and tensors to free GPU/CPU resources.

For further discussion and community troubleshooting, see the Unity Discussions groups on Sentis. [Discussions](https://discussions.unity.com/t/feedback-on-sentis-2-1-0/1552905)

---

## Conclusion

Sentis 2.1.2 provides a powerful and flexible framework for running neural networks in Unity. By understanding its core concepts, properly managing tensors and workers, and utilizing the Functional API for model customization, developers can integrate advanced machine learning capabilities into their Unity projects efficiently. This documentation—complemented by the extensive samples and API references—serves as a solid foundation for both beginners and advanced users to harness runtime AI within Unity.

For additional resources and the latest updates, always refer to the official Unity documentation and sample repositories. 

---

This document has drawn on multiple sources from the official Unity documentation and community samples. For more details, see:
- [Sentis overview and manual](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/index.html)
- [Sentis Samples](https://docs.unity3d.com/Packages/com.unity.sentis%402.1/manual/package-samples.html)
- [Scripting API reference](https://docs.unity.cn/Packages/com.unity.sentis%402.1/api/Unity.Sentis.html)

Happy coding and experimentation with runtime AI in Unity!
