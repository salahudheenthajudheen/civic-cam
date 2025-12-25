import React, { useRef, useEffect } from 'react';

function FaceDetectionCanvas({ imageRef, faceBox, isSuspectFace = false }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    
    if (!canvas || !img) return;

    const drawFaceDetection = () => {
      const ctx = canvas.getContext('2d');
      
      // Set canvas size to match image
      if (img.offsetWidth && img.offsetHeight) {
        canvas.width = img.offsetWidth;
        canvas.height = img.offsetHeight;
      } else if (img.complete && img.naturalWidth) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      } else {
        return;
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (isSuspectFace) {
        // For suspect face, draw box around the entire face area
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, canvas.width, canvas.height);
      } else {
        // For main image, draw box at detected position
        if (faceBox && img.complete && img.naturalWidth) {
          const scaleX = canvas.width / img.naturalWidth;
          const scaleY = canvas.height / img.naturalHeight;
          
          ctx.strokeStyle = '#00ff00';
          ctx.lineWidth = 3;
          ctx.strokeRect(
            faceBox.x * scaleX,
            faceBox.y * scaleY,
            faceBox.width * scaleX,
            faceBox.height * scaleY
          );
        }
      }
    };

    // Draw when image loads
    const handleImageLoad = () => {
      // Small delay to ensure image dimensions are set
      setTimeout(drawFaceDetection, 10);
    };

    if (img.complete && img.naturalWidth > 0) {
      handleImageLoad();
    } else {
      img.addEventListener('load', handleImageLoad);
    }

    // Handle window resize
    const handleResize = () => {
      setTimeout(drawFaceDetection, 10);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      img.removeEventListener('load', handleImageLoad);
      window.removeEventListener('resize', handleResize);
    };
  }, [imageRef, faceBox, isSuspectFace]);

  return (
    <canvas 
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}
    />
  );
}

export default FaceDetectionCanvas;

