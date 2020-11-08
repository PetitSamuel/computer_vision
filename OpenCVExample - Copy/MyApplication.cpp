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
	for (int chan = 0; chan < image.channels(); chan++) {
		calcHist(&(colour_channels[chan]), 1, channel_numbers, Mat(),
			histogram[chan], 1, &number_bins, &channel_ranges);
		normalize(histogram[chan], histogram[chan], 1.0);
	}
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

vector<Mat> findForegroundMask(Mat& src_background, Mat& temp_selective_running_average_background, Mat& processed_img) {
	// Find Foreground mask
	Mat selective_running_average_difference;
	vector<Mat> selective_running_average_planes(3);
	src_background.convertTo(temp_selective_running_average_background, CV_8U);
	absdiff(temp_selective_running_average_background, processed_img, selective_running_average_difference);
	split(selective_running_average_difference, selective_running_average_planes);
	return selective_running_average_planes;
}

void updateAvgSelectiveBackground(vector<Mat> input_planes, vector<Mat> selective_running_average_planes, Mat& mask_output) {
	double running_average_learning_rate = 0.01;
	accumulateWeighted(input_planes[0], selective_running_average_planes[0], running_average_learning_rate, mask_output);
	accumulateWeighted(input_planes[1], selective_running_average_planes[1], running_average_learning_rate, mask_output);
	accumulateWeighted(input_planes[2], selective_running_average_planes[2], running_average_learning_rate, mask_output);
	invertImage(mask_output, mask_output);
}

double computeCmpHist(MatND* base_histo, MatND* cmp_histo, int method, int nb_chan) {
	double matching_score = 0.0;
	for (int i = 0; i < nb_chan; i++) {
		matching_score = matching_score + compareHist(base_histo[0], cmp_histo[0], method);
	}
	return matching_score;
}

void computeSingleBoxMask(Mat& src, Mat& dst, int postbox_index) {
	Point corners[] = {
	 Point(PostboxLocations[postbox_index][POSTBOX_TOP_LEFT_COLUMN],PostboxLocations[postbox_index][POSTBOX_TOP_LEFT_ROW]),
	 Point(PostboxLocations[postbox_index][POSTBOX_BOTTOM_LEFT_COLUMN],PostboxLocations[postbox_index][POSTBOX_BOTTOM_LEFT_ROW]),
	 Point(PostboxLocations[postbox_index][POSTBOX_BOTTOM_RIGHT_COLUMN],PostboxLocations[postbox_index][POSTBOX_BOTTOM_RIGHT_ROW]),
	 Point(PostboxLocations[postbox_index][POSTBOX_TOP_RIGHT_COLUMN],PostboxLocations[postbox_index][POSTBOX_TOP_RIGHT_ROW]),
	};
	const Point* corner_list[] = { corners };
	int num_points = 4;
	int num_polygons = 1;
	Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	fillPoly(mask, corner_list, &num_points, num_polygons, cv::Scalar(255, 255, 255));
	Mat result(src.size(), src.type(), cv::Scalar(255, 255, 255));
	src.copyTo(result, mask);
	dst = result.clone();
}

void checkBoxesHavePost(Mat& src, Mat& background_src, bool box_has_post[]) {
	for (int i = 0; i < NUMBER_OF_POSTBOXES; i++) {
		// Mask the current box from the current foreground and first img foreground.
		Mat single_postbox_img, single_postobox_background;
		computeSingleBoxMask(src, single_postbox_img, i);
		computeSingleBoxMask(background_src, single_postobox_background, i);
		// Compute histograms for both
		Mat histo_img, hist_b_img;
		MatND* single_postbox_histo = computeHistogram(single_postbox_img, histo_img);
		MatND* single_postobox_background_histo = computeHistogram(single_postobox_background, hist_b_img);
		// Compare both histograms.
		int method = cv::HISTCMP_CHISQR;
		double matching_sum = computeCmpHist(single_postobox_background_histo, single_postbox_histo, method, src.channels());
		if (matching_sum >= 2) {
			box_has_post[i] = true;
		}
	}
}

void makeOutputString(bool box_has_post[], int frame_count) {
	char output_str[100] = "";
	bool found_a_letter = false;
	for (int i = 0; i < NUMBER_OF_POSTBOXES; i++) {
		if (box_has_post[i]) {
			int true_index = i + 1;
			found_a_letter = true;
			char to_append[20];
			sprintf(to_append, " %d", true_index);
			strcat(output_str, to_append);
		}
	}
	if (!found_a_letter) {
		cout << frame_count << ", " << "No post" << "\n";
		return;
	}
	cout << frame_count << ", Post in" << output_str << "\n";
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
	int starting_frame = 0;

	if (!video.isOpened()) {
		cout << "Video file is not open: " << filename << endl;
		return;
	}

	Mat current_frame, selective_running_average_background, selective_running_average_foreground_image, first_frame_foreground;
	double running_average_learning_rate = 0.01;
	video.set(cv::CAP_PROP_POS_FRAMES, starting_frame);
	video >> current_frame;
	Mat& first_frame = current_frame.clone();

	// Grab first frame for selective avg background.
	applyProcessingToFrame(current_frame, selective_running_average_background);
	selective_running_average_background.convertTo(selective_running_average_background, CV_32F);

	MatND* first_frame_histo;

	int frame_count = 1;
	while ((!current_frame.empty()))
	{
		Mat processed_img = current_frame.clone();
		applyProcessingToFrame(current_frame, processed_img, true);	

		Mat temp_selective_running_average_background;
		vector<Mat> selective_running_average_planes = findForegroundMask(selective_running_average_background, temp_selective_running_average_background, processed_img);
		
		// Sum up points and run Otsu thresholding to obtain the foreground.
		// TODO compare with all /3 and use 30 cst thresh
		Mat temp_sum = (selective_running_average_planes[0] + selective_running_average_planes[1] + selective_running_average_planes[2]);
		Mat selective_running_average_foreground_mask;
		threshold(temp_sum, selective_running_average_foreground_mask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
		// Run a close operation on the result.
		Mat closed_image;
		morphologyEx(selective_running_average_foreground_mask, selective_running_average_foreground_mask, MORPH_CLOSE, Mat());
		/*
		TODO compare both techinques see if 1 is better.
		Mat opened_image;
		morphologyEx(selective_running_average_foreground_mask, opened_image, MORPH_OPEN, Mat());
		*/

		// Update background
		vector<Mat> input_planes(3);
		split(processed_img, input_planes);
		split(selective_running_average_background, selective_running_average_planes);
		updateAvgSelectiveBackground(input_planes, selective_running_average_planes, selective_running_average_foreground_mask);

		// Copy the pixels from the current frame into the foreground image using the selecting running average foreground mask.
		selective_running_average_foreground_image.setTo(Scalar(0, 0, 0));
		current_frame.copyTo(selective_running_average_foreground_image, selective_running_average_foreground_mask);
		
		Mat histo_img;
		MatND* colour_histogram = computeHistogram(selective_running_average_foreground_image, histo_img);
		if (frame_count == 1) {
			first_frame_foreground = selective_running_average_foreground_image.clone();
			first_frame_histo = computeHistogram(selective_running_average_foreground_image, Mat());
		}

		int method = cv::HISTCMP_CHISQR;
		double matching_sum = computeCmpHist(first_frame_histo, colour_histogram, method, selective_running_average_foreground_image.channels());

		if (matching_sum > 2) {
			cout << frame_count << ", View obscured" << "\n";
		}
		else {
			bool box_has_post[NUMBER_OF_POSTBOXES] = {
				false, false, false, false, false, false
			};
			checkBoxesHavePost(selective_running_average_foreground_image, first_frame_foreground, box_has_post);
			makeOutputString(box_has_post, frame_count);
		}

		char frame_str[100];
		sprintf(frame_str, "Frame = %d, cmp = %f", frame_count++, matching_sum);
		Mat temp_selective_output = JoinImagesHorizontally(current_frame, frame_str, temp_selective_running_average_background, "Selective Running Average Background", 4);
		Mat selective_output = JoinImagesHorizontally(temp_selective_output, "", selective_running_average_foreground_image, "Foreground", 4);
		imshow("Selective Running Average Background Model", selective_output);
		video >> current_frame;
		char c = cv::waitKey(1000);
	}
	cv::destroyAllWindows();

}
