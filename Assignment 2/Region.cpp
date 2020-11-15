/*
* This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
* by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
*/
#include "Utilities.h"


Mat kmeans_clustering(Mat& image, int k, int iterations)
{
	CV_Assert(image.type() == CV_8UC3);
	// Populate an n*3 array of float for each of the n pixels in the image
	Mat samples(image.rows*image.cols, image.channels(), CV_32F);
	float* sample = samples.ptr<float>(0);
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				samples.at<float>(row*image.cols + col, channel) =
				(uchar)image.at<Vec3b>(row, col)[channel];
	// Apply k-means clustering to cluster all the samples so that each sample
	// is given a label and each label corresponds to a cluster with a particular
	// centre.
	Mat labels;
	Mat centres;
	kmeans(samples, k, labels, TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 1, 0.0001),
		iterations, KMEANS_PP_CENTERS, centres);
	// Put the relevant cluster centre values into a result image
	Mat& result_image = Mat(image.size(), image.type());
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				result_image.at<Vec3b>(row, col)[channel] = (uchar)centres.at<float>(*(labels.ptr<int>(row*image.cols + col)), channel);
	return result_image;
}

// This routine colors the regions
// Code taken from Open Source meanshift_segmentation.cpp (widely available online)
void floodFillPostprocess(Mat& img, const Scalar& colorDiff /* = Scalar::all(1)*/)
{
	CV_Assert(!img.empty());
	RNG rng = theRNG();
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)
			{
				Scalar newVal(rng(256), rng(256), rng(256));
				floodFill(img, mask, Point(x, y), newVal, 0, colorDiff, colorDiff);
			}
		}
	}
}

void RegionDemos(Mat& image1, Mat& image2, Mat& image3)
{
	Mat gray_image1,binary_image, gray_image3;
	cvtColor(image1, gray_image1, COLOR_BGR2GRAY);
	cvtColor(image3, gray_image3, COLOR_BGR2GRAY);

	// Grey scale morphology + Connected Components Analysis
//	morphologyEx(gray_pcb_image, opened_image, MORPH_OPEN, five_by_five_element);
	threshold(gray_image1, binary_image, 135, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat binary_image_copy = binary_image.clone();
	findContours(binary_image_copy, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	Mat contours_image = Mat::zeros(binary_image.size(), CV_8UC3);
	for (int contour_number = 0; (contour_number<(int)contours.size()); contour_number++)
	{
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(contours_image, contours, contour_number, colour, cv::FILLED, 8, hierarchy);
	}
	Mat binary_display,output1,output2,output3;
	cvtColor(binary_image, binary_display, COLOR_GRAY2BGR);
	output1 = JoinImagesHorizontally(image1, "Original Image", binary_image, "Binary image", 4);
	output2 = JoinImagesHorizontally(binary_display, "Thresholded Image", contours_image, "Connected Components", 4);
	imshow("Grey Scale Morphology & Connected Components", output2);
	char c = cv::waitKey();
	cv::destroyAllWindows();

	// K-means clustering
	Mat clustered_image = kmeans_clustering(image3, 15, 5);
	output1 = JoinImagesHorizontally(image3, "Original Image", clustered_image, "k-Means Clustered Image", 4);
	imshow("k-Means Clustering", output1);
	c = cv::waitKey();
	cv::destroyAllWindows();

	// Mean shift clustering/segmentation
	pyrMeanShiftFiltering(image2, clustered_image, 40, 40, 2);
	floodFillPostprocess(clustered_image, Scalar::all(2));
	output1 = JoinImagesHorizontally(image2, "Original Image", clustered_image, "Mean Shift (40,40)", 4);
	imshow("Mean Shift Segmentation", output1);
	c = cv::waitKey();
	cv::destroyAllWindows();

}
