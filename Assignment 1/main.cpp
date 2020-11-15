#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

// Method which takes a number of white pixels
// returns the amount of expected spoons.
int PixelsToSpoonsAmount(int pixels) {
	if (pixels > 10000) {
		return 2;
	}
	else if (pixels > 2000) {
		return 1;
	}
	return 0;
}

void ThresholdAndOpen(Mat& image)
{
	Mat otsu_binary_image;
	// Lighting is very similar on all images so I use constant thresholding.
	threshold(image, otsu_binary_image, 175, 255, THRESH_BINARY);
	// Do an open operation on the image.
	Mat opened_image;
	morphologyEx(otsu_binary_image, opened_image, MORPH_OPEN, Mat());
	image = opened_image;
}

void SaturationImage(Mat& image)
{
	// Convert image to HSL
	Mat hls_image;
	cvtColor(image, hls_image, COLOR_BGR2HLS);
	// Split into its 3 planes.
	vector<Mat> hls_planes(3);
	split(hls_image, hls_planes);
	// Keep saturation channel.
	image = hls_planes[2];
}

int main(int argc, const char** argv)
{
	char* file_location = "Media/";
	char* image_files[] = {
		"BabyFood-Sample0.jpg", //0
		"BabyFood-Sample1.jpg",
		"BabyFood-Sample2.jpg",
		"BabyFood-Test1.jpg",
		"BabyFood-Test2.jpg",
		"BabyFood-Test3.jpg",
		"BabyFood-Test4.jpg",
	    "BabyFood-Test5.jpg", //5
	    "BabyFood-Test6.jpg",
	    "BabyFood-Test7.jpg",
	    "BabyFood-Test8.jpg",
		"BabyFood-Test9.jpg",
		"BabyFood-Test10.jpg", //10
		"BabyFood-Test11.jpg",
		"BabyFood-Test12.jpg",
		"BabyFood-Test13.jpg",
		"BabyFood-Test14.jpg",
		"BabyFood-Test15.jpg", //15
		"BabyFood-Test16.jpg",
		"BabyFood-Test17.jpg" ,
		"BabyFood-Test18.jpg",
    };

	// Load images
	int number_of_images = sizeof(image_files) / sizeof(image_files[0]);

	Mat* image = new Mat[number_of_images];
	for (int file_no=0; (file_no < number_of_images); file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		image[file_no] = imread(filename, -1);
		if (image[file_no].empty())
		{
			cout << "Could not open " << filename << endl;
			return -1;
		}
	}

	Point location( 7, 25);
	Scalar colour( 0, 0, 255);
	for (int i = 0; i < number_of_images; i++) {
		Mat current_image = image[i].clone();
		// Extract saturation from image.
		SaturationImage(current_image);
		// Threshold and open.
		ThresholdAndOpen(current_image);
		// Retrieve number of white pixels.
		int pixels = cv::countNonZero(current_image);
		// Use the number to estimate a number of spoons in the image.
		int number_spoons = PixelsToSpoonsAmount(pixels);
		// Convert the final processed image to color.
		Mat out_img;
		cvtColor(current_image, out_img, COLOR_GRAY2BGR);
		// Add number of found spoons to the image.
		putText(out_img, std::to_string(number_spoons) + " spoons found.", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
		// Show output.
		Mat output = JoinImagesHorizontally(image[i], "Original Image (press any key to move to the next image)", out_img, "Processed image", 4);
		imshow("output", output);
		cv::waitKey();
		cv::destroyAllWindows();
	}
}
