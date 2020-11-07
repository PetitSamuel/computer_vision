#include "Utilities.h"
#include "opencv2/video.hpp"

// Assignment vars
#define POSTBOX_VIDEO_INDEX 0
#define NUMBER_OF_POSTBOXES 6
int PostboxLocations[NUMBER_OF_POSTBOXES][8] = {
	{ 26, 113, 106, 113, 13, 133, 107, 134 },
	{ 119, 115, 199, 115, 119, 135, 210, 136 },
	{ 30, 218, 108, 218, 18, 255, 109, 254 },
	{ 119, 217, 194, 217, 118, 253, 207, 253 },
	{ 32, 317, 106, 315, 22, 365, 108, 363 },
	{ 119, 315, 191, 314, 118, 362, 202, 361 } };
#define POSTBOX_TOP_LEFT_COLUMN 0
#define POSTBOX_TOP_LEFT_ROW 1
#define POSTBOX_TOP_RIGHT_COLUMN 2
#define POSTBOX_TOP_RIGHT_ROW 3
#define POSTBOX_BOTTOM_LEFT_COLUMN 4
#define POSTBOX_BOTTOM_LEFT_ROW 5
#define POSTBOX_BOTTOM_RIGHT_COLUMN 6
#define POSTBOX_BOTTOM_RIGHT_ROW 7

MatND* computeHistogram(Mat& image, Mat& histo_img) {
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 64;
	MatND* histogram = new MatND[image.channels()];
	vector<Mat> colour_channels(image.channels());
	split(image, colour_channels);
	for (int chan = 0; chan < image.channels(); chan++)
		calcHist(&(colour_channels[chan]), 1, channel_numbers, Mat(),
			histogram[chan], 1, &number_bins, &channel_ranges);
	DrawHistogram(histogram, image.channels(), histo_img);
	return histogram;
}

void applyProcessingToFrame(Mat& src, Mat& dst, bool applyBlur = false) {
	src.convertTo(dst, -1, 1, -100); //decrease the brightness by 100
	if (applyBlur) {
		addGaussianNoise(dst, 0, 20.0);
		GaussianBlur(dst, dst, Size(5, 5), 3);
	}
}

void MyApplication()
{
	// Load video(s)
	char* video_files = "PostboxesWithLines.avi";
	char* file_location = "Media/";
	VideoCapture* videos = new VideoCapture[1];
	string filename(file_location);
	filename.append(video_files);
	videos[POSTBOX_VIDEO_INDEX].open(filename);
	if (!videos[POSTBOX_VIDEO_INDEX].isOpened())
	{
		cout << "Cannot open video file: " << filename << endl;
		return;
	}

	VideoCapture& video = videos[POSTBOX_VIDEO_INDEX];
	int starting_frame = 1;

	if (!video.isOpened()) {
		cout << "Video file is not open: " << filename << endl;
		return;
	}

	Mat current_frame, selective_running_average_background,
		temp_selective_running_average_background, selective_running_average_difference,
		selective_running_average_foreground_mask, selective_running_average_foreground_image;
	double running_average_learning_rate = 0.01;
	video.set(cv::CAP_PROP_POS_FRAMES, starting_frame);
	video >> current_frame;
	Mat& first_frame = current_frame.clone();

	// Grab first frame for selective avg background.
	applyProcessingToFrame(current_frame, selective_running_average_background);
	selective_running_average_background.convertTo(selective_running_average_background, CV_32F);

	double frame_rate = video.get(cv::CAP_PROP_FPS);
	double time_between_frames = 1000.0 / frame_rate;
	int frame_count = 0;
	while ((!current_frame.empty()))
	{
		Mat processed_img = current_frame.clone();
		applyProcessingToFrame(current_frame, processed_img, true);	
		// Running Average with selective update
		vector<Mat> selective_running_average_planes(3);
		// Find Foreground mask
		selective_running_average_background.convertTo(temp_selective_running_average_background, CV_8U);
		absdiff(temp_selective_running_average_background, processed_img, selective_running_average_difference);
		split(selective_running_average_difference, selective_running_average_planes);
		// Determine foreground points as any point with an average difference of more than 30 over all channels:
		Mat temp_sum = (selective_running_average_planes[0] / 3 + selective_running_average_planes[1] / 3 + selective_running_average_planes[2] / 3);
		threshold(temp_sum, selective_running_average_foreground_mask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
		imshow("Test tresh", selective_running_average_foreground_mask);
		Mat opened_image;
		morphologyEx(selective_running_average_foreground_mask, opened_image, MORPH_OPEN, Mat());
		imshow("after open", opened_image);
		selective_running_average_foreground_mask = opened_image;

		// updateAvgSelectiveBackground();
		// Update background
		vector<Mat> input_planes(3);
		split(processed_img, input_planes);
		split(selective_running_average_background, selective_running_average_planes);
		accumulateWeighted(input_planes[0], selective_running_average_planes[0], running_average_learning_rate, selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[1], selective_running_average_planes[1], running_average_learning_rate, selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[2], selective_running_average_planes[2], running_average_learning_rate, selective_running_average_foreground_mask);
		invertImage(selective_running_average_foreground_mask, selective_running_average_foreground_mask);

		selective_running_average_foreground_image.setTo(Scalar(0, 0, 0));
		current_frame.copyTo(selective_running_average_foreground_image, selective_running_average_foreground_mask);
		Mat histo_img;
		MatND* colour_histogram = computeHistogram(selective_running_average_foreground_image, histo_img);
		imshow("histo", histo_img);

		char c = cv::waitKey(250);

		char frame_str[100];
		sprintf(frame_str, "Frame = %d", frame_count);
		Mat temp_selective_output = JoinImagesHorizontally(current_frame, frame_str, temp_selective_running_average_background, "Selective Running Average Background", 4);
		Mat selective_output = JoinImagesHorizontally(temp_selective_output, "", selective_running_average_foreground_image, "Foreground", 4);
		imshow("Selective Running Average Background Model", selective_output);
		video >> current_frame;
	}
	cv::destroyAllWindows();

}
