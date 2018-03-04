clc;
clear all;
close all;

%%
% Reads Video.
videoSequence = VideoReader('walk_qcif.avi');

% Stores first 10 frames.
count = 1;
while (hasFrame(videoSequence) && count <11);
    frameNumber(count).cdata = readFrame(videoSequence);         
    count = count+1;
end

originalFrameCopy = frameNumber;  % Keeps copy of Original Frames.

%%
% Intialize size for database and stores matrices which will finally be copied to database for transmission.

encoderBufferZigzagLumaAC = cell(1,5);
encoderBufferDifferentialLumaDC = cell(1,5);
encoderBufferZigzagCbAC = cell(1,5);
encoderBufferDifferentialCbDC = cell(1,5);
encoderBufferZigzagCrAC = cell(1,5);
encoderBufferDifferentialCrDC = cell(1,5);
encoderBufferMVCurrX = cell(1,5);
encoderBufferMVCurrY = cell(1,5);
encoderBufferIndicationMatrix = cell(1,5);

%% Encoder.

for number = 6:(count-1);

    % Displays Original Current Frame at Encoder in RGB format.
    figure();
    imshow(frameNumber(number).cdata);
    title(['Original Current Frame ' int2str(number) ' at ENCODER in RGB Format']); 
    
    % Converts RGB to YCbCr.
    currentFrameYCbCr = rgb2ycbcr(frameNumber(number).cdata);
    currentFrameY = currentFrameYCbCr(:,:,1); % Extracts Luminance Component.
    currentFrameCb = currentFrameYCbCr(:,:,2); % Extracts Chrominance(Cb) Component.
    currentFrameCr = currentFrameYCbCr(:,:,3); % Extracts Chrominance(Cr) Component.
    
    % Size of each Frame.
    [row, column] = size(currentFrameY);
    
    % 4:2:0 Downsampling.    
    subSampledYCurrentFrame = currentFrameY;
    subSampledCbCurrentFrame = currentFrameCb(1:2:end, 1:2:end);
    subSampledCrCurrentFrame = currentFrameCr(1:2:end, 1:2:end);
    
    % Intializes Motion Vector Matrix to store the values.
    motionVectorXValue = zeros(row/16,column/16,'double');    
    motionVectorYValue = zeros(row/16,column/16,'double');    
    
    % Initializes Difference Frames for Luma and Chroma components.
    differenceFrameLuma = zeros(row, column, 'double');
    differenceFrameCb = zeros(row/2, column/2, 'double');
    differenceFrameCr = zeros(row/2, column/2, 'double');
       
    % Initializes Indication matrix.
    indicationMatrix = zeros(row/16,column/16,'double');
 
    % Initializes Luma and Chroma components for Reference Frame.
    subSampledYReferenceFrame = zeros(row, column, 'double');
    subSampledCbReferenceFrame = zeros(row/2, column/2, 'double');
    subSampledCrReferenceFrame = zeros(row/2, column/2, 'double');
        
    %% Motion Estimation.
    % If Current Frame is not I frame, we need to find MV and Difference Matrix.
    
    if(number ~= 6)

        % Displays Original Reconstructed Reference Frame at Encoder in RGB format.
        figure();
        imshow(frameNumber(number-1).cdata);
        title(['Reference Frame ' int2str(number-1) ' at ENCODER in RGB Format']);
        
        % Converts RGB to YCbCr.
        referenceFrameYCbCr = rgb2ycbcr(frameNumber(number-1).cdata);
        referenceFrameY = referenceFrameYCbCr(:,:,1); % Extracts Luminance Component.
        referenceFrameCb = referenceFrameYCbCr(:,:,2); % Extracts Chrominance(Cb) Component.
        referenceFrameCr = referenceFrameYCbCr(:,:,3); % Extracts Chrominance(Cr) Component.
        
        % 4:2:0 Downsampling.
        subSampledYReferenceFrame = referenceFrameY;
        subSampledCbReferenceFrame = referenceFrameCb(1:2:end, 1:2:end);
        subSampledCrReferenceFrame = referenceFrameCr(1:2:end, 1:2:end);
        
        % Extracts Reference and Current Frame Blocks for Motion Estimation.
        for p = 0:((row/16)-1);
           for q = 0:((column/16)-1);

               currentMacroBlock = double(currentFrameY((p*16)+1:(p+1)*16, (q*16)+1:(q+1)*16));

               % Creates Search Window of suitable size.

               if((((q*16)+1)-8)<=0)
                   startColumnSearchWindow = 1;
               else
                   startColumnSearchWindow = (((q*16)+1)-8);
               end

               if((((p*16)+1)-8)<=0)
                   startRowSearchWindow = 1;
               else
                   startRowSearchWindow = (((p*16)+1)-8);
               end

               if(((q+1)*16)+8 >= column)
                   endColumnSearchWindow = column;
               else
                   endColumnSearchWindow = ((q+1)*16)+8;
               end

               if(((p+1)*16)+8 >= row)    
                   endRowSearchWindow = row;
               else
                   endRowSearchWindow = ((p+1)*16)+8;
               end

               referenceSW = double(referenceFrameY(startRowSearchWindow : endRowSearchWindow, startColumnSearchWindow : endColumnSearchWindow));

               % Stores relative X, Y values and difference MB.
               [ xShift, yShift, differenceFrameLuma((p*16)+1:(p+1)*16, (q*16)+1:(q+1)*16)] = exhaustiveSearchAlgorithm( currentMacroBlock, referenceSW);

               % Calculates Motion Vector.
               motionVectorXValue(p+1,q+1) = (p*16 + 1) - (xShift - 1 + startRowSearchWindow);
               motionVectorYValue(p+1,q+1) = (q*16 + 1) - (yShift - 1 + startColumnSearchWindow);

           end
        end
        
        figure;
        imshow(uint8(differenceFrameLuma));
        title(['Luma Component of Difference frame for Current Frame No. ' int2str(number+1) ' at ENCODER']);

        % Plots Motion Vector.
        [x,y] = meshgrid(1:((column/16)),1:((row/16)));
        figure;
        quiver(x,y, motionVectorXValue, motionVectorYValue);
        title(['Motion Vector for Frame No. ' int2str(number+1)]);

        % Creates Difference Matrix for Chroma components.
        for p=0:(row/16)-1;
            for q=0:(column/16)-1;
                movementXChroma = floor(motionVectorXValue(p+1,q+1)/2);
                movementYChroma = floor(motionVectorYValue(p+1,q+1)/2);
                differenceFrameCb((p*8)+1:(p+1)*8, (q*8)+1:(q+1)*8) = double(subSampledCbCurrentFrame((p*8)+1:(p+1)*8, (q*8)+1:(q+1)*8)) - double(subSampledCbReferenceFrame(((p*8)+1-movementXChroma):(((p+1)*8)-movementXChroma), ((q*8)+1-movementYChroma):(((q+1)*8)-movementYChroma)));
                differenceFrameCr((p*8)+1:(p+1)*8, (q*8)+1:(q+1)*8) = double(subSampledCrCurrentFrame((p*8)+1:(p+1)*8, (q*8)+1:(q+1)*8)) - double(subSampledCrReferenceFrame(((p*8)+1-movementXChroma):(((p+1)*8)-movementXChroma), ((q*8)+1-movementYChroma):(((q+1)*8)-movementYChroma)));
            end
        end
    end     
        
    % If Current Frame is I frame, our differenceFrame* matrix will be zero. 
    % So we need to copy subSampled*CurrentFrame matrix value to them. 
    
    if(number==6)        
        differenceFrameLuma = double(subSampledYCurrentFrame);
        differenceFrameCb = double(subSampledCbCurrentFrame);
        differenceFrameCr = double(subSampledCrCurrentFrame); 
    end
        
    % Difference Matrix and MV for Current Frame are derived.
    
    %% Discrete Cosine Transform (DCT).

    [rowY, colY] = size(differenceFrameLuma);
    [rowCbCr,colCbCr] = size(differenceFrameCb);
    
    % DCT works on 8x8 matrix. Thus padding is required if size is not a multiple of 8x8.
    padderY = zeros(mod(rowY,8),mod(colY,8));
    padderCbCr = zeros(mod(rowCbCr,8),mod(colCbCr,8));

    % Padding in case size is not a multiple of 8x8 or 16x16.
    YDCTMatrix = [differenceFrameLuma; double(padderY)];
    CbDCTMatrix = [differenceFrameCb; double(padderCbCr)];
    CrDCTMatrix = [differenceFrameCr; double(padderCbCr)];

    % Size of padded matrix.
    [rowPaddedY, colPaddedY] = size(YDCTMatrix);
    [rowPaddedCbCr, colPaddedCbCr] = size(CbDCTMatrix);
    
    % Performs DCT for Luma component, processing 8x8 blocks at a time.
    for m = 0:((rowPaddedY/8)-1);
       for n = 0:((colPaddedY/8)-1);
          YDCTCurrentFrame((m*8)+1:(m+1)*8, (n*8)+1:(n+1)*8) = dct2(YDCTMatrix((m*8)+1:(m+1)*8, (n*8)+1:(n+1)*8));
       end
    end
    % pDCT = @dct2;
    % YDCTCurrentFrame = blkproc (YDCTMatrix, [8 8], pDCT); % Calculates DCT for Luma component

    % Performs DCT for Chroma components, processing 8x8 blocks at a time.
    for m = 0:((rowPaddedCbCr/8)-1);
       for n = 0:((colPaddedCbCr/8)-1);
           CbDCTCurrentFrame((m*8)+1:(m+1)*8, (n*8)+1:(n+1)*8) = dct2(CbDCTMatrix((m*8)+1:(m+1)*8, (n*8)+1:(n+1)*8));
           CrDCTCurrentFrame((m*8)+1:(m+1)*8, (n*8)+1:(n+1)*8) = dct2(CrDCTMatrix((m*8)+1:(m+1)*8, (n*8)+1:(n+1)*8));
       end
    end
    % CbDCTCurrentFrame = blkproc (CbDCTMatrix, [8 8], pDCT);
    % CrDCTCurrentFrame = blkproc (CrDCTMatrix, [8 8], pDCT);
    
    %% Quantization.
    quantizationValue = 28;
    YQuantized = round(YDCTCurrentFrame/quantizationValue);
    CbQuantized = round(CbDCTCurrentFrame/quantizationValue);
    CrQuantized = round(CrDCTCurrentFrame/quantizationValue);
    
    %% Zig-Zag Scan.
    
    % Performs Zig-Zag for difference Luma and Chroma components.
    % Passing 2nd argument in zigZag function helps in prevention of 
    % sending those MBs which have MV (0,0) and SAD less than 128.
    
    [currentLumaDC, zigZagCurrentLumaAC] = zigZag(YQuantized, indicationMatrix, 2);            
    [currentCbDC, zigZagCurrentCbAC] = zigZag(CbQuantized, indicationMatrix, 1);
    [currentCrDC, zigZagCurrentCrAC] = zigZag(CrQuantized, indicationMatrix, 1);
    
    %% Calculates Differential DC coefficients.
    
    differentialCurrentLumaDC = differentialCoding(currentLumaDC);
    differentialCurrentCbDC = differentialCoding(currentCbDC);
    differentialCurrentCrDC = differentialCoding(currentCrDC); 
  
    %% Saves the parameters for Transmission.
    
    encoderBufferZigzagLumaAC{1,number-5} = zigZagCurrentLumaAC;
    encoderBufferDifferentialLumaDC{1,number-5} = differentialCurrentLumaDC;
    encoderBufferZigzagCbAC{1,number-5} = zigZagCurrentCbAC;
    encoderBufferDifferentialCbDC{1,number-5} = differentialCurrentCbDC;
    encoderBufferZigzagCrAC{1,number-5} = zigZagCurrentCrAC;
    encoderBufferDifferentialCrDC{1,number-5} = differentialCurrentCrDC;
    encoderBufferMVCurrX{1,number-5} = motionVectorXValue;
    encoderBufferMVCurrY{1,number-5} = motionVectorYValue;
    encoderBufferIndicationMatrix{1,number-5} = indicationMatrix;
    
    %% Inbuild Decoder.
    
    [reconstructedCurrentLuma, reconstructedCurrentCb, reconstructedCurrentCr] = inbuildDecoder(YQuantized, CbQuantized, CrQuantized, motionVectorXValue, motionVectorYValue, subSampledYReferenceFrame, subSampledCbReferenceFrame, subSampledCrReferenceFrame, indicationMatrix);
    
    %% Upsampling using Row Column Replication.
    
    % Luma does not need upsampling.
    upsampledCurrentLuma = reconstructedCurrentLuma;
    
    % Upsamples Cb and Cr components using Row Column Replication.
    upsampledCurrentCb = zeros (row, column, 'uint8');  %Creates zeros matrix for Cb component.
    upsampledCurrentCr = zeros (row, column, 'uint8');  %Creates zeros matrix for Cr component.
    
    upsampledCurrentCb(1:2:end,1:2:end)=reconstructedCurrentCb(1:end,1:end);
    upsampledCurrentCb(2:2:end,:)=upsampledCurrentCb(1:2:end,:);
    upsampledCurrentCb(:,2:2:end)=upsampledCurrentCb(:,1:2:end);
    
    upsampledCurrentCr(1:2:end,1:2:end)= reconstructedCurrentCr(1:end,1:end);
    upsampledCurrentCr(2:2:end,:)=upsampledCurrentCr(1:2:end,:);
    upsampledCurrentCr(:,2:2:end)=upsampledCurrentCr(:,1:2:end);
    
    % Concatenates for YCbCr image.
    reconstructedYCbCr = cat(3, upsampledCurrentLuma, upsampledCurrentCb, upsampledCurrentCr);
    
    % Converts YCbCr to RGB format.
    reconstructedRGB = ycbcr2rgb(reconstructedYCbCr);
    
    % Displays reconstructed current frame at Encoder.
    figure();
    imshow(reconstructedRGB);
    title(['Reconstructed Current RGB frame ' int2str(number) ' at ENCODER']);
       
    %% Peak Signal to Noise Ratio.
    
    Y_Error = upsampledCurrentLuma - currentFrameY; % Error in Luma component.

    MSEY = sum(sum(Y_Error.^2))/(row*column); % Calculates MSE for Luma component.

    PSNRY = 10*log10((255.^2)/MSEY); % Calculates PSNR.

    display(['Calculated PSNR for Luma component of Current Frame no. ' int2str(number) ' is = ' num2str(PSNRY)]);
    
    
    %% Refills the Database.
    
    frameNumber(number).cdata(:,:,1) = reconstructedRGB(:,:,1);  
    frameNumber(number).cdata(:,:,2) = reconstructedRGB(:,:,2);  
    frameNumber(number).cdata(:,:,3) = reconstructedRGB(:,:,3);  
    
end

%% Intializes Database for Transmission.

database = cell(1,9);

database{1,1} = encoderBufferZigzagLumaAC;
database{1,2} = encoderBufferDifferentialLumaDC;
database{1,3} = encoderBufferZigzagCbAC;
database{1,4} = encoderBufferDifferentialCbDC;
database{1,5} = encoderBufferZigzagCrAC;
database{1,6} = encoderBufferDifferentialCrDC;
database{1,7} = encoderBufferMVCurrX;
database{1,8} = encoderBufferMVCurrY;
database{1,9} = encoderBufferIndicationMatrix;

%% Decoder.

MCSHomework4_Decoder(database);
