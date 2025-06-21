# LiveTalk-Unity

Unity package for using LiveTalk on-device models for real-time talking head generation.

## What is LiveTalk?

LiveTalk is an open-source real-time high quality talking head generation system that can generate natural-looking talking head videos from avatar images and audio input. This Unity package makes it easy to incorporate this technology into your Unity projects for AI-driven character animation.

## Key Features:

* 🎮 **Unity-Native Integration**: Simple API designed specifically for Unity
* 🎭 **Fast Avatar Animation**: Generate talking head videos
* 👤 **Multi-Avatar Support**: Process multiple avatar images for seamless transitions
* 💻 **Runs Offline**: All processing happens on-device
* ⚡ **Optimized Performance**: Memory-efficient pipeline with caching and quantization support

## Perfect For:

* AI-driven NPCs in games
* Virtual assistants and chatbots
* Real-time character animation
* Interactive storytelling applications
* Video content generation
* Accessibility features

## Installation

### Using Unity Package Manager (Recommended)

1. Open your Unity project
2. Open the Package Manager (Window > Package Manager)
3. Click the "+" button in the top-left corner
4. Select "Add package from git URL..."
5. Enter the repository URL: `https://github.com/arghyasur1991/LiveTalk-Unity.git`
6. Click "Add"

### Manual Installation

1. Clone this repository
2. Copy the contents into your Unity project's Packages folder

## Dependencies

This package requires the following Unity packages:
- com.github.asus4.onnxruntime (0.4.0)
- com.github.asus4.onnxruntime-extensions (0.4.0)

### Setting up Package Dependencies

Some dependencies require additional scoped registry configuration. Add the following to your project's `Packages/manifest.json` file:

```json
{
  "scopedRegistries": [
    {
      "name": "NPM",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    "com.genesis.LiveTalk.unity": "file:/path/to/LiveTalk-Unity",
    // ... other dependencies
  }
}
```

**Note**: Replace `/path/to/LiveTalk-Unity` with the actual path to your LiveTalk-Unity package folder.

## Features

- Real-time talking head generation from avatar images and audio
- WIP

## Usage

### Basic Talking Head Generation

```csharp
using UnityEngine;
using LiveTalk.Core;
using LiveTalk.Models;
using System.Threading.Tasks;

public class LiveTalkExample : MonoBehaviour
{
    // WIP
}
```

### Advanced Configuration

```csharp
using LiveTalk.Models;

// Create configuration for LiveTalk
var config = new LiveTalkConfig
{
    ModelPath = "path/to/onnx/models",
    Version = "v15", // or "v13"
    UseINT8 = true, // Enable INT8 quantization for better performance
    PreferINT8Models = true,
    BboxShift = 0
};

// Initialize with custom configuration
var LiveTalk = new LiveTalkInference(config);
```

### Performance Monitoring

```csharp
// Enable detailed performance monitoring
LiveTalkInference.EnablePerformanceMonitoring = true;
LiveTalkInference.LogTiming = true;

// Check quantization status
Debug.Log($"Using quantization: {LiveTalk.QuantizationMode}");
Debug.Log($"INT8 enabled: {LiveTalk.IsUsingINT8}");
```

## Model Setup

This package requires LiveTalk ONNX models in the following location:

```
Assets/StreamingAssets/LiveTalk/
  ├── WIP
  ├── WIP
  └── WIP
```

### INT8 Quantized Models (Optional)

For better performance, you can also include INT8 quantized versions:

```
Assets/StreamingAssets/LiveTalk/
  ├── WIP
  └── ... (other INT8 models)
```

### Exporting Models

You can obtain these models by using the conversion scripts from the original LiveTalk repository and converting them to ONNX format using :

1. WIP

### Pre-Exported ONNX Models

Pre-exported ONNX models will be made available for download. Check the releases section of this repository.

## Configuration Options

### LiveTalkConfig Properties

- **ModelPath**: Path to the ONNX models directory
- **UseINT8**: Enable INT8 quantization for performance
- **PreferINT8Models**: Prefer INT8 models when available
- **BboxShift**: Adjustment for face bounding box detection

### Performance Tuning

- Use INT8 quantization for CPU optimization
- Adjust batch size based on available memory
- Enable performance monitoring for optimization
- Use avatar animation caching for repeated sequences

## Requirements

- Unity 6000.0 or later
- Platforms: MacOS (CPU only), Windows (Not tested)
- Minimum _GB RAM recommended

## Performance

Typical performance on modern hardware:
- **Mac M4 Max (CPU)**: 1-2 FPS

## Troubleshooting

### Common Issues

1. **Models not found**: Ensure ONNX models are in StreamingAssets/LiveTalk/
2. **Out of memory**: Reduce batch size or enable INT8 quantization
3. **Slow performance**: Enable INT8 models
4. **Face detection fails**: Ensure input images contain clear, frontal faces

## License

WIP

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## Credits

- InsightFace integration for face processing
- ONNX Runtime for model inference
- Unity ML integration and optimization

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes. 