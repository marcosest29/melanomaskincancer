#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;
Mat paso1(Mat img_rgb){
	Mat img_hsv,eqv,hsveq,out;
	cvtColor(img_rgb,img_hsv,CV_RGB2HSV);
	vector<cv::Mat> channels;
	split(img_hsv, channels);
	equalizeHist(channels[2], eqv );
	channels[2]=eqv;
	merge(channels,hsveq);
	cvtColor(hsveq,out,CV_HSV2RGB);
	cvtColor(out,out,CV_RGB2GRAY);
	Mat image_blurred_with_21x21_kernel;
    	GaussianBlur(out, image_blurred_with_11x11_kernel, Size(11, 11), 0);
	Mat binary;
	threshold( out, binary, 57, max_BINARY_value,1 );
    	return binary;}

void generateHistogram(Mat image, int histogram[])
{
	//initialize all intesity values to 0
	for (int i=0;i<256;i++)
		histogram[i]=0;
	//Compute the number of pixels for each intensity values
	for (int row=0; row<image.rows; row++)
		for (int col=0; col<image.cols; col++)
			histogram[(int)image.at<uchar>(row,col)]++;
	
}

void displayHistogram(int histogram[], const char* windowName)
{
	int histTmp[256];
	for (int i=0; i<256; i++)
		histTmp[i]=histogram[i];

	//Draw the histogram, define width and height
	int hist_w =500; int hist_h=500;
	int bin_w = cvRound( (double) hist_w/256);

	//Create the Image of Histogram
	cv::Mat histImage(hist_h, hist_w, CV_8UC1, Scalar (255,255,255));

	//find maximum intensity element from histogram
	int max=histTmp[0];
	for (int i =1; i<256; i++)
		if (max <histTmp[i])
			max=histTmp[i];

	//Normalize he histgram between 0 and histImage.rows
	for (int i=0; i<256; i++)
		histTmp[i] = ( (double) histTmp[i]/max) * histImage.rows;

	//Draw the intensiy line for frequency in the histogram
	for (int i=0; i<256; i++)
		line(histImage, Point(bin_w * i, hist_h),
				    Point(bin_w * i , hist_h-histTmp[i]), 
				    Scalar(0,0,0),
				    1,8,0);

	//DISPLAY HISTOGRAM
 	namedWindow( windowName, cv::WINDOW_NORMAL );
	resizeWindow( windowName, 450,450);
  	imshow( windowName, histImage );

}

void cumHistogram(int histogram[], int cumhistogram[])
{
	cumhistogram[0]=histogram[0];
	for (int i=1; i<256; i++)
		cumhistogram[i]=histogram[i]+cumhistogram[i-1];
}
 
int main( int argc, char** argv ) 
{
  
	Mat image;
	image = cv::imread("/home/andres/Escritorio/ImageProce/Lab8/dataset/melanoma-skin-cancer.jpg" , CV_LOAD_IMAGE_GRAYSCALE);
	Mat image1= imread("/home/andres/Escritorio/ImageProce/Lab8/dataset/melanoma-skin-cancer.jpg", CV_LOAD_IMAGE_COLOR);
    
	if(!image.data or !image1.data) 
	{
		std::cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
  
//	namedWindow( "Original image", cv::WINDOW_NORMAL );
//	resizeWindow("Original image", 450,450);
//	imshow( "Original image", image );

	int histogram[256];
	generateHistogram(image, histogram);

	// Display the original Histogram
	displayHistogram(histogram, "Original Histogram");


	int w,h,ihist[256],pos,OPT_TH;
	float hist_val[256],p1K,meanitr_K,meanitr_G,param1,param2,param3;
	h=image.rows;
	w=image.cols;  
	generateHistogram(image,ihist);
	p1K=0;meanitr_K=0;meanitr_G=0;OPT_TH=0;
	//Normalize histogram
	for(int i=0;i<256;i++)
	{
		hist_val[i]=(float)(ihist[i])/(float)(w*h);
		meanitr_G+=(float)(i*hist_val[i]);
	}
	
	//OTSU
	Mat imageRes;
	for(int i=0;i<255;i++)
	{
		p1K+=(float)hist_val[i];
		meanitr_K+=(float)(i * hist_val[i]);
		param1 = (float)(meanitr_G * p1K) - meanitr_K;
		param2 = (float)(param1*param1)/(float)p1K*(1-p1K);
        	if(param2>param3)
		{
			param3=param2;
			OPT_TH=i;
        	}
      	}	
	threshold(image,imageRes,OPT_TH,255,0);
	
	//Canny(imageRes,imageRes,20,100,3);	
	namedWindow("image",WINDOW_NORMAL);
	resizeWindow("image",450,450);
	imshow("image",imageRes);
	namedWindow("Original",WINDOW_NORMAL);
	resizeWindow("Original",450,450);
	imshow("Original",image1);
  	waitKey(0);
  	return 0;
}
