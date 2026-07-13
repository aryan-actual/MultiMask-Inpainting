import { useRef, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';
import axios from 'axios';

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [brushRadius, setBrushRadius] = useState<number>(20);
  const [useFastModel, setUseFastModel] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [resultUrl, setResultUrl] = useState<string>('');
  const [imageSize, setImageSize] = useState<{width: number, height: number}>({width: 512, height: 512});

  const canvasRef = useRef<CanvasDraw | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setResultUrl('');

      // Get image dimensions to size the canvas
      const img = new Image();
      img.onload = () => {
        // Limit max width/height to 512 or reasonable size for UI
        let { width, height } = img;
        const maxDim = 800;
        if (width > maxDim || height > maxDim) {
          const ratio = Math.min(maxDim / width, maxDim / height);
          width = width * ratio;
          height = height * ratio;
        }
        setImageSize({ width, height });
      };
      img.src = url;
    }
  };

  // Helper to convert transparent canvas to black/white mask
  const generateMask = async (): Promise<Blob | null> => {
    if (!canvasRef.current) return null;
    
    // Get the drawing as data URL (it has transparent background)
    // Wait, the types don't officially expose getDataURL without arguments sometimes, but we can access the underlying canvas.
    // @ts-ignore
    // react-canvas-draw's getDataURL doesn't always preserve pure transparency or exact stroke thickness over custom backgrounds properly
    // It's safer to extract it and force binary coloring on the frontend before sending
    const drawingDataUrl = canvasRef.current.getDataURL('png', false, '#000000');
    
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = imageSize.width;
        canvas.height = imageSize.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve(null);

        // Fill black background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw the strokes (which will have whatever color they had in getDataURL)
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Force to strictly binary white and black pixels based on drawn strokes
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {
          // If the pixel is not purely black (R>0, G>0, or B>0), force it to pure white
          if (data[i] > 10 || data[i+1] > 10 || data[i+2] > 10) {
            data[i] = 255;     // R
            data[i+1] = 255;   // G
            data[i+2] = 255;   // B
            data[i+3] = 255;   // Alpha
          } else {
            data[i] = 0;
            data[i+1] = 0;
            data[i+2] = 0;
            data[i+3] = 255;
          }
        }
        ctx.putImageData(imgData, 0, 0);

        canvas.toBlob((blob) => {
          resolve(blob);
        }, 'image/png');
      };
      img.src = drawingDataUrl;
    });
  };

  const handleSubmit = async () => {
    if (!imageFile) return alert("Please upload an image first.");
    if (!prompt) return alert("Please enter a prompt.");
    
    setIsLoading(true);
    
    try {
      const maskBlob = await generateMask();
      if (!maskBlob) throw new Error("Failed to generate mask");

      const formData = new FormData();
      formData.append('image', imageFile);
      // maskBlob needs a filename
      formData.append('mask', maskBlob, 'mask.png');
      formData.append('prompt', prompt);

      const endpoint = useFastModel ? 'http://localhost:8000/inpaint-fast' : 'http://localhost:8000/inpaint';
      const response = await axios.post(endpoint, formData, {
        responseType: 'blob'
      });

      const outUrl = URL.createObjectURL(response.data);
      setResultUrl(outUrl);
    } catch (err: any) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || "Error generating image.";
      alert(`Error: ${errorMsg}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 text-gray-800">
      <div className="max-w-5xl mx-auto bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">Qwen Image Inpainting</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* Controls */}
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Upload Base Image</label>
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleImageUpload}
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>

            {imageUrl && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Brush Size: {brushRadius}</label>
                  <input 
                    type="range" 
                    min="5" 
                    max="100" 
                    value={brushRadius}
                    onChange={(e) => setBrushRadius(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <button 
                    onClick={() => canvasRef.current?.clear()} 
                    className="mt-2 text-sm text-red-500 hover:underline"
                  >
                    Clear Mask
                  </button>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Prompt</label>
                  <textarea 
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full border border-gray-300 rounded p-2"
                    rows={3}
                    placeholder="Describe what you want to generate in the masked area..."
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <input 
                    type="checkbox" 
                    id="fastMode" 
                    checked={useFastModel} 
                    onChange={(e) => setUseFastModel(e.target.checked)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="fastMode" className="text-sm font-medium text-gray-700">
                    Use Fast Mode (ControlNet + Lightning 4-Steps)
                  </label>
                </div>

                <button 
                  onClick={handleSubmit} 
                  disabled={isLoading}
                  className="w-full bg-blue-600 text-white font-bold py-3 rounded shadow hover:bg-blue-700 disabled:opacity-50"
                >
                  {isLoading ? 'Generating...' : 'Generate Edit'}
                </button>
              </>
            )}
          </div>

          {/* Canvas & Preview */}
          <div className="flex flex-col items-center justify-center bg-gray-100 rounded border border-gray-200 overflow-hidden min-h-[400px]">
            {!imageUrl && <p className="text-gray-500">Upload an image to start masking</p>}
            
            {imageUrl && !resultUrl && (
              <div className="relative" style={{ width: imageSize.width, height: imageSize.height }}>
                <img 
                  src={imageUrl} 
                  alt="Base" 
                  className="absolute top-0 left-0"
                  style={{ width: imageSize.width, height: imageSize.height }}
                />
                <div className="absolute top-0 left-0 opacity-70">
                  <CanvasDraw
                    ref={canvasRef}
                    canvasWidth={imageSize.width}
                    canvasHeight={imageSize.height}
                    brushRadius={brushRadius}
                    lazyRadius={0}
                    brushColor="#ffffff"
                    hideGrid={true}
                    backgroundColor="transparent"
                  />
                </div>
              </div>
            )}

            {resultUrl && (
              <div className="flex flex-col items-center w-full h-full p-4">
                <h3 className="font-semibold mb-2">Result:</h3>
                <img src={resultUrl} alt="Result" className="max-w-full max-h-[600px] object-contain rounded shadow" />
                <button 
                  onClick={() => setResultUrl('')}
                  className="mt-4 text-blue-600 hover:underline"
                >
                  Edit Again
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
