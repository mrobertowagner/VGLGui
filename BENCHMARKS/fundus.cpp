
#ifdef __OPENCL__
//this program requires opencl

#include "vglClImage.h"
#include "vglContext.h"
#include "cl2cpp_shaders.h"
#include "glsl2cpp_shaders.h"
#include "vglClFunctions.h"

#ifdef __OPENCV__
  #include <opencv2/imgproc/types_c.h>
  #include <opencv2/imgproc/imgproc_c.h>
  #include <opencv2/highgui/highgui_c.h>
#else
  #include <vglOpencv.h>
#endif

#include "demo/timer.h"

#include <fstream>
#include <string.h>


int main(int argc, char* argv[])
{


    vglClInit();

    int nSteps = atoi(argv[2]);
    char* inFilename = argv[1];
    char* outPath = argv[3];
    char* outFilename = (char*) malloc(strlen(outPath) + 200);

    printf("VisionGL-OpenCL on %s, %d operations\n\n", inFilename, nSteps);
	
    printf("CREATING IMAGE\n");
    
    VglImage* img = vglLoadImage(inFilename, CV_LOAD_IMAGE_UNCHANGED, 0);
    
    if (img->nChannels == 3)
    {
        printf("NCHANNELS = 3\n");
        if (img->ndarray)
        {
            printf("NDARRAY not null\n");
            vglNdarray3To4Channels(img);
        }
        else
        {
            printf("NDARRAY IS null\n");
            vglIpl3To4Channels(img);
        }
    }
    
    // Convolution kernels
    float kernel33[3][3]    = { {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
                                {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
                                {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}, }; //blur 3x3
    float kernel55[5][5]    = { {1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f},
                                {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
                                {6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f},
                                {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
                                {1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f}, }; //blur 5x5
    float erodeMask51[51] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    float kernel1[1][51] = {0.00037832,0.00055477,0.00080091,0.00113832,0.00159279,0.00219416,0.00297573,0.00397312,0.00522256,0.0067585,0.00861055,0.01080005,0.01333629,0.0162128,0.01940418,0.02286371,0.02652237,0.0302895,0.0340554,0.03769589,0.04107865,0.04407096,0.04654821,0.04840248,0.04955031,0.04993894,0.04955031,0.04840248,0.04654821,0.04407096,0.04107865,0.03769589,0.0340554,0.0302895,0.02652237,0.02286371,0.01940418,0.0162128,0.01333629,0.01080005,0.00861055,0.0067585,0.00522256,0.00397312,0.00297573,0.00219416,0.00159279,0.00113832,0.00080091,0.00055477,0.00037832};
    float kernel51[51][1] = {0.00037832,0.00055477,0.00080091,0.00113832,0.00159279,0.00219416,0.00297573,0.00397312,0.00522256,0.0067585,0.00861055,0.01080005,0.01333629,0.0162128,0.01940418,0.02286371,0.02652237,0.0302895,0.0340554,0.03769589,0.04107865,0.04407096,0.04654821,0.04840248,0.04955031,0.04993894,0.04955031,0.04840248,0.04654821,0.04407096,0.04107865,0.03769589,0.0340554,0.0302895,0.02652237,0.02286371,0.01940418,0.0162128,0.01333629,0.01080005,0.00861055,0.0067585,0.00522256,0.00397312,0.00297573,0.00219416,0.00159279,0.00113832,0.00080091,0.00055477,0.00037832};

   float erodeMask1717[289] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};


    VglImage* gray = vglCreateImage(img);
    int p = 0;
    //First call to Convert to gray
    TimerStart();
    vglClRgbToGray(img, gray);
    vglClFlush();
    printf("First call to          Convert to Gray: %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Convert to gray
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClRgbToGray(img, gray);
    }
    vglClFlush();
    printf("Time spent on %8d Convert to Gray: %s\n", nSteps, getTimeElapsedInSeconds());

    vglCheckContext(gray, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/gray.tif");
    cvSaveImage(outFilename, gray->ipl);

    //----------------------------------------------------------
    // CONVOLUTION 1 X 51
    VglImage* conv = vglCreateImage(img);
    TimerStart();

    vglClConvolution(gray, conv, (float*) kernel1, 1, 51);

    vglClFlush();
    printf("First call to          Convolution 1x51:         %s\n", getTimeElapsedInSeconds());
    
    //Total time spent on n operations Convolution 1x51
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClConvolution(gray, conv, (float*) kernel1, 1, 51);
    }
    vglClFlush();
    printf("Time spent on %8d Convolution 1x51:         %s \n", nSteps, getTimeElapsedInSeconds());


    //CONVOLUTION 51 X 1
    VglImage* conv_out = vglCreateImage(conv);
    TimerStart();
    vglClConvolution(conv, conv_out, (float*) kernel51, 51, 1);
    vglClFlush();
    printf("First call to          Convolution 51x1:         %s\n", getTimeElapsedInSeconds());

    //Total time spent on n operations Convolution 51x1
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClConvolution(conv, conv_out, (float*) kernel51, 51, 1);
    }
    vglClFlush();
    printf("Time spent on %8d Convolution 51x1:         %s \n", nSteps, getTimeElapsedInSeconds());


    

    vglCheckContext(conv_out, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/conv.tif");
    cvSaveImage(outFilename, conv_out->ipl);
    
    //----------------------------------------------------------
    //DILATE 1 X 51
    VglImage* dil = vglCreateImage(conv_out);
    TimerStart();

    vglClDilate(conv_out, dil, erodeMask51, 1, 51);
    vglClFlush();

    printf("Fisrt call to          Dilate1x51:                  %s\n", getTimeElapsedInSeconds());
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClDilate(conv_out, dil, erodeMask51, 1, 51);
    }
    vglClFlush();
    printf("Time spent on %8d Dilate1x51:                  %s\n", nSteps, getTimeElapsedInSeconds());

    
    //DILATE 51 X 1 
    VglImage* dil_out = vglCreateImage(dil);
    TimerStart();

    vglClDilate(dil, dil_out, erodeMask51, 51, 1);
    vglClFlush();
    printf("Fisrt call to          Dilate51x1:                  %s\n", getTimeElapsedInSeconds());
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClDilate(dil, dil_out, erodeMask51, 51, 1);
    }
    vglClFlush();
    printf("Time spent on %8d Dilate51x1:                  %s\n", nSteps, getTimeElapsedInSeconds());
    

    vglCheckContext(dil_out, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/dil.tif");
    cvSaveImage(outFilename, dil_out->ipl);

    //----------------------------------------------------------
    //ERODE 1 X 51
    VglImage* erod = vglCreateImage(dil_out);
    TimerStart();

    vglClErode(dil_out, erod, erodeMask51, 1, 51);
    vglClFlush();
    printf("First call to          Erode 1x51:               %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Erode 1x51
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClErode(dil_out, erod, erodeMask51, 1, 51);
    }
    vglClFlush();
    printf("Time spent on %8d Erode 1x51:               %s\n", nSteps, getTimeElapsedInSeconds());



    
    //ERODE 51 X 1
    VglImage* erod_out = vglCreateImage(erod);
    TimerStart();

    vglClErode(erod, erod_out, erodeMask51, 51, 1);
    vglClFlush();
    printf("First call to          Erode 51x1:               %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Erode 51x1
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClErode(erod, erod_out, erodeMask51, 51, 1);
    }
    vglClFlush();
    printf("Time spent on %8d Erode 51x1:               %s\n", nSteps, getTimeElapsedInSeconds());

    vglCheckContext(erod_out, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/erod.tif");
    cvSaveImage(outFilename, erod_out->ipl);

    //----------------------------------------------------------

    // SUB (ERODE - CONV)
    VglImage* sub = vglCreateImage(erod_out);
    TimerStart();

    vglClSub(erod_out, conv_out, sub);
    vglClFlush();

    printf("First call to          Sub:               %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Sub
    p = 0;
    TimerStart();
    while (p< nSteps)
    {
        p++;
        vglClSub(erod_out, conv_out, sub);
    }
    vglClFlush();
    printf("Time spent on %8d Sub:              %s\n", nSteps, getTimeElapsedInSeconds());
    vglCheckContext(sub, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/sub.tif");
    cvSaveImage(outFilename, sub->ipl);


    //----------------------------------------------------------

    VglImage* thresh = vglCreateImage(sub);
    TimerStart();

    vglClThreshold(sub, thresh, 0.011);
    vglClFlush();
    printf("First call to          Theshold:               %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Thresh
    p = 0;
    TimerStart();
    while (p< nSteps)
    {
        p++;
        vglClThreshold(sub, thresh, 0.00784);
        
    }
    vglClFlush();
    printf("Time spent on %8d Threshold:              %s\n", nSteps, getTimeElapsedInSeconds());

    vglCheckContext(thresh, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/thresh.tif");
    cvSaveImage(outFilename, thresh->ipl);

    //----------------------------------------------------------

    VglImage* eroderec = vglCreateImage(thresh);
    
    
    VglImage* buffer = vglCreateImage(thresh);


    TimerStart();
    vglClErode(thresh, eroderec, erodeMask1717, 17, 17);

    
    VglImage* rec_out = vglCreateImage(eroderec);

    vglClReconstructionByDilation(eroderec, eroderec, rec_out,buffer, erodeMask1717, 17, 17);
    vglClFlush();
    printf("First call to          Reconstruction:               %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Reconstruction

    p = 0;
    TimerStart();
    while (p< nSteps)
    {
        p++;
        vglClErode(thresh, eroderec, erodeMask1717, 17, 17);
        vglClReconstructionByDilation(eroderec, eroderec, rec_out,buffer, erodeMask1717, 17, 17);
        
    }
    vglClFlush();
    printf("Time spent on %8d Reconstruction:              %s\n", nSteps, getTimeElapsedInSeconds());
   
    vglCheckContext(rec_out, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/rec.tif");
    cvSaveImage(outFilename, rec_out->ipl);


    //flush
    vglClFlush();
    return 0;

}

#endif
