
#ifdef __OPENCL__
//this program requires opencl

#include "vglClImage.h"
#include "vglContext.h"
#include "cl2cpp_shaders.h"
#include "glsl2cpp_shaders.h"

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

    if (argc != 4)
    {
        printf("\nUsage: demo_benchmark_cl lena_1024.tiff 1000 /tmp\n\n");
        printf("In this example, will run the program for lena_1024.tiff in a \nloop with 1000 iterations. Output images will be stored in /tmp.\n\n");
        printf("Error: Bad number of arguments = %d. 3 arguments required.\n", argc-1);
        exit(1);
    }
    //vglInit(500,500);
    vglClInit();

    int nSteps = atoi(argv[2]);
    char* inFilename = argv[1];
    char* outPath = argv[3];
    char* outFilename = (char*) malloc(strlen(outPath) + 200);

    printf("VisionGL-OpenCL on %s, %d operations\n\n", inFilename, nSteps);
	
    printf("CREATING IMAGE\n");
    VglImage* img = vglLoadImage(inFilename, CV_LOAD_IMAGE_UNCHANGED, 0);
 

    printf("CHECKING NCHANNELS\n");
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

    img->vglShape->print();
    iplPrintImageInfo(img->ipl);

    printf("CHECKING IF IS NULL\n");
    if (img == NULL)
    {
        std::string str("Error: File not found: ");
        str.append(inFilename);
        printf("%s", str.c_str());
    }

    printf("CREATING COPY\n");



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
    sprintf(outFilename, "%s%s", outPath, "/out_cl_gray.tif");
    cvSaveImage(outFilename, gray->ipl);

    VglImage* conv = vglCreateImage(gray);

    // Convolution kernels
    float kernel33[3][3]    = { {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
                                {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
                                {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}, }; //blur 3x3
    float kernel55[5][5]    = { {1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f},
                                {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
                                {6.0f/256.0f, 24.0f/256.0f, 36.0f/256.0f, 24.0f/256.0f, 6.0f/256.0f},
                                {4.0f/256.0f, 16.0f/256.0f, 24.0f/256.0f, 16.0f/256.0f, 4.0f/256.0f},
                                {1.0f/256.0f,  4.0f/256.0f,  6.0f/256.0f,  4.0f/256.0f, 1.0f/256.0f}, }; //blur 5x5
    float kernel151[1][51] = {0.00037832,0.00055477,0.00080091,0.00113832,0.00159279,0.00219416,0.00297573,0.00397312,0.00522256,0.0067585,0.00861055,0.01080005,0.01333629,0.0162128,0.01940418,0.02286371,0.02652237,0.0302895,0.0340554,0.03769589,0.04107865,0.04407096,0.04654821,0.04840248,0.04955031,0.04993894,0.04955031,0.04840248,0.04654821,0.04407096,0.04107865,0.03769589,0.0340554,0.0302895,0.02652237,0.02286371,0.01940418,0.0162128,0.01333629,0.01080005,0.00861055,0.0067585,0.00522256,0.00397312,0.00297573,0.00219416,0.00159279,0.00113832,0.00080091,0.00055477,0.00037832};


    //First call to Convolution 3x3
    TimerStart();
    vglClConvolution(gray, conv, (float*) kernel151, 5, 5);
    vglClFlush();
    printf("First call to          Convolution 5x5:         %s\n", getTimeElapsedInSeconds());

    //Total time spent on n operations Convolution 3x3
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClConvolution(gray, conv, (float*) kernel55, 5, 5);
    }
    vglClFlush();
    printf("Time spent on %8d Convolution 5x5:         %s \n", nSteps, getTimeElapsedInSeconds());


    vglCheckContext(conv, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_conv.tif");
    cvSaveImage(outFilename, conv->ipl);


    VglImage* erod = vglCreateImage(conv);
    //First call to Erode 3x3
    float erodeMask[9] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    float erodeMask25[25] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

    TimerStart();
    vglClErode(conv, erod, erodeMask25, 5, 5);
    vglClFlush();
    printf("First call to          Erode 5x5:               %s \n", getTimeElapsedInSeconds());
    //Total time spent on n operations Erode 3x3
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClErode(conv, erod, erodeMask25, 5, 5);
    }
    vglClFlush();
    printf("Time spent on %8d Erode 5x5:               %s\n", nSteps, getTimeElapsedInSeconds());

    vglCheckContext(erod, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_erosion.tif");
    cvSaveImage(outFilename, erod->ipl);

    VglImage* dil = vglCreateImage(erod);
    //First call to Dilate
    TimerStart();
    vglClDilate(erod, dil, erodeMask25, 5, 5);
    vglClFlush();
    printf("Fisrt call to          Dilate5x5:                  %s\n", getTimeElapsedInSeconds());
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClDilate(erod, dil, erodeMask25, 5, 5);
    }
    vglClFlush();
    printf("Time spent on %8d Dilate5x5:                  %s\n", nSteps, getTimeElapsedInSeconds());

    vglCheckContext(dil, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_dilate.tif");
    cvSaveImage(outFilename, dil->ipl);
  

    //flush
    vglClFlush();
    return 0;

}

#endif
