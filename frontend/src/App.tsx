import { useRef, useState, createRef } from 'react';
import CanvasDraw from 'react-canvas-draw';
import axios from 'axios';

interface Step {
  id: number;
  prompt: string;
  referenceFile: File | null;
}

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  
  const [steps, setSteps] = useState<Step[]>([{ id: Date.now(), prompt: '', referenceFile: null }]);
  const [activeStepIndex, setActiveStepIndex] = useState<number>(0);
  
  const [brushRadius, setBrushRadius] = useState<number>(20);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [resultUrl, setResultUrl] = useState<string>('');
  const [imageSize, setImageSize] = useState<{width: number, height: number}>({width: 512, height: 512});

  const MASK_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7"];

  // We support up to 5 steps, so we pre-create 5 refs
  const canvasRefs = useRef((Array(5).fill(0).map(() => createRef<CanvasDraw>())));

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setResultUrl('');
      setSteps([{ id: Date.now(), prompt: '', referenceFile: null }]);
      setActiveStepIndex(0);

      // Get image dimensions to size the canvas
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        const maxWidth = 450;
        const maxHeight = 500;
        if (width > maxWidth || height > maxHeight) {
          const ratio = Math.min(maxWidth / width, maxHeight / height);
          width = width * ratio;
          height = height * ratio;
        }
        setImageSize({ width, height });
      };
      img.src = url;
    }
  };

  const handleAddStep = () => {
    if (steps.length >= 5) return;
    setSteps([...steps, { id: Date.now(), prompt: '', referenceFile: null }]);
    setActiveStepIndex(steps.length);
  };

  const handleRemoveStep = (indexToRemove: number) => {
    if (steps.length <= 1) return;
    const newSteps = steps.filter((_, i) => i !== indexToRemove);
    setSteps(newSteps);
    if (activeStepIndex >= newSteps.length) {
      setActiveStepIndex(newSteps.length - 1);
    }
  };

  const handlePromptChange = (index: number, val: string) => {
    const newSteps = [...steps];
    newSteps[index].prompt = val;
    setSteps(newSteps);
  };

  const handleReferenceUpload = (index: number, e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const newSteps = [...steps];
      newSteps[index].referenceFile = e.target.files[0];
      setSteps(newSteps);
    }
  };
  
  const handleRemoveReference = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const newSteps = [...steps];
    newSteps[index].referenceFile = null;
    setSteps(newSteps);
  }

  const generateMask = async (index: number): Promise<Blob | null> => {
    const canvasRef = canvasRefs.current[index];
    if (!canvasRef.current) return null;
    
    // @ts-ignore
    const drawingDataUrl = canvasRef.current.getDataURL('png', false, '#000000');
    
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = imageSize.width;
        canvas.height = imageSize.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return resolve(null);

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] > 10 || data[i+1] > 10 || data[i+2] > 10) {
            data[i] = 255;
            data[i+1] = 255;
            data[i+2] = 255;
            data[i+3] = 255;
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
    
    // Validate that all steps have a prompt
    const invalidStep = steps.findIndex(s => !s.prompt.trim());
    if (invalidStep !== -1) {
      setActiveStepIndex(invalidStep);
      return alert(`Please enter a prompt for Step ${invalidStep + 1}.`);
    }
    
    setIsLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('original_image', imageFile);

      const editsJson = [];

      // Generate mask and append for each step
      for (let i = 0; i < steps.length; i++) {
        const maskBlob = await generateMask(i);
        if (!maskBlob) throw new Error(`Failed to generate mask for step ${i + 1}`);
        
        const maskFilename = `mask_${i}.png`;
        formData.append('files', maskBlob, maskFilename);
        
        let refFilename = null;
        if (steps[i].referenceFile) {
          refFilename = `ref_${i}_${steps[i].referenceFile?.name}`;
          formData.append('files', steps[i].referenceFile as File, refFilename);
        }

        editsJson.push({
          mask: maskFilename,
          prompt: steps[i].prompt,
          reference: refFilename
        });
      }

      formData.append('edits', JSON.stringify(editsJson));

      const API_BASE = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;
      const endpoint = `${API_BASE}/edit`;
      
      const response = await axios.post(endpoint, formData, {
        responseType: 'blob'
      });

      const outUrl = URL.createObjectURL(response.data);
      setResultUrl(outUrl);
    } catch (err: any) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || err.message || "Error generating image.";
      alert(`Error: ${errorMsg}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 text-gray-800">
      <div className="max-w-5xl mx-auto bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">FLUX.1 Kontext Sequential Editing</h1>
        
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
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <h3 className="font-semibold mb-3">Masking Steps ({steps.length}/5)</h3>
                  <div className="space-y-4">
                    {steps.map((step, index) => (
                      <div 
                        key={step.id}
                        className={`p-3 rounded-lg border transition-colors cursor-pointer ${activeStepIndex === index ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500' : 'border-gray-300 bg-white hover:border-blue-300'}`}
                        onClick={() => setActiveStepIndex(index)}
                      >
                        <div className="flex justify-between items-center mb-2">
                          <label className="text-sm font-semibold text-gray-800 flex items-center">
                            <span 
                              className="w-3 h-3 rounded-full mr-2 inline-block" 
                              style={{ backgroundColor: MASK_COLORS[index % MASK_COLORS.length] }}
                            ></span>
                            Prompt for Mask {index + 1}
                          </label>
                          {index === activeStepIndex && (
                            <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded-full font-medium">
                              Drawing...
                            </span>
                          )}
                        </div>
                        <textarea 
                          value={step.prompt}
                          onChange={(e) => handlePromptChange(index, e.target.value)}
                          onClick={() => setActiveStepIndex(index)}
                          className="w-full border border-gray-300 rounded p-2 text-sm bg-white mb-2"
                          rows={2}
                          placeholder={`Describe what to generate in mask ${index + 1}...`}
                        />
                        
                        <div className="mb-2">
                           <label className="block text-xs font-medium text-gray-700 mb-1">Optional Reference Image</label>
                           {!step.referenceFile ? (
                              <input 
                                type="file" 
                                accept="image/*" 
                                onChange={(e) => handleReferenceUpload(index, e)}
                                onClick={() => setActiveStepIndex(index)}
                                className="w-full text-xs border border-gray-300 rounded p-1 bg-white"
                             />
                           ) : (
                             <div className="flex items-center justify-between bg-blue-100 p-2 rounded text-xs font-medium">
                                <span className="truncate">{step.referenceFile.name}</span>
                                <button onClick={(e) => handleRemoveReference(index, e)} className="text-red-500 hover:underline">Remove</button>
                             </div>
                           )}
                        </div>
                        
                        {index === activeStepIndex && (
                          <div className="mt-2 flex justify-between items-center">
                            <button 
                              onClick={(e) => { e.stopPropagation(); canvasRefs.current[index].current?.clear(); }} 
                              className="text-xs text-red-500 hover:underline"
                            >
                              Clear Mask {index + 1}
                            </button>

                            {steps.length > 1 && (
                              <button 
                                onClick={(e) => { e.stopPropagation(); handleRemoveStep(index); }}
                                className="text-xs text-red-600 hover:underline font-medium"
                              >
                                Remove Step
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                    
                    {steps.length < 5 && (
                      <button 
                        onClick={handleAddStep}
                        className="w-full py-2 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 font-medium hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50 transition-colors"
                      >
                        + Add Another Mask
                      </button>
                    )}
                  </div>
                </div>

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
                </div>

                <button 
                  onClick={handleSubmit} 
                  disabled={isLoading}
                  className="w-full bg-blue-600 text-white font-bold py-3 rounded shadow hover:bg-blue-700 disabled:opacity-50 mt-4"
                >
                  {isLoading ? 'Generating (This may take a while)...' : 'Generate Edit'}
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
                
                {/* Render up to 5 stacked canvases, show all of them but only active one accepts drawing */}
                {steps.map((step, index) => (
                  <div 
                    key={step.id}
                    className="absolute top-0 left-0"
                    style={{ 
                      display: 'block',
                      opacity: index === activeStepIndex ? 0.8 : 0.4,
                      zIndex: index === activeStepIndex ? 10 : index,
                      pointerEvents: index === activeStepIndex ? 'auto' : 'none'
                    }}
                  >
                    <CanvasDraw
                      ref={canvasRefs.current[index]}
                      canvasWidth={imageSize.width}
                      canvasHeight={imageSize.height}
                      brushRadius={brushRadius}
                      lazyRadius={0}
                      brushColor={MASK_COLORS[index % MASK_COLORS.length]}
                      hideGrid={true}
                      backgroundColor="transparent"
                    />
                  </div>
                ))}
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
