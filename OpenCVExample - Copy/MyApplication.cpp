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


class MedianBackgroundTest
{
private:
	Mat mMedianBackgroundTest;
	float**** mHistogram;
	float*** mLessThanMedian;
	float mAgingRate;
	float mCurrentAge;
	float mTotalAges;
	int mValuesPerBin;
	int mNumberOfBins;
public:
	MedianBackgroundTest(Mat initial_image, float aging_rate, int values_per_bin);
	Mat GetBackgroundImage();
	void UpdateBackground(Mat current_frame);
	float getAgingRate()
	{
		return mAgingRate;
	}
};

MedianBackgroundTest::MedianBackgroundTest(Mat initial_image, float aging_rate, int values_per_bin)
{
	mCurrentAge = 1.0;
	mAgingRate = aging_rate;
	mTotalAges = 0.0;
	mValuesPerBin = values_per_bin;
	mNumberOfBins = 256 / mValuesPerBin;
	mMedianBackgroundTest = Mat::zeros(initial_image.size(), initial_image.type());
	mLessThanMedian = (float***) new float** [mMedianBackgroundTest.rows];
	mHistogram = (float****) new float*** [mMedianBackgroundTest.rows];
	for (int row = 0; (row < mMedianBackgroundTest.rows); row++)
	{
		mHistogram[row] = (float***) new float** [mMedianBackgroundTest.cols];
		mLessThanMedian[row] = (float**) new float* [mMedianBackgroundTest.cols];
		for (int col = 0; (col < mMedianBackgroundTest.cols); col++)
		{
			mHistogram[row][col] = (float**) new float* [mMedianBackgroundTest.channels()];
			mLessThanMedian[row][col] = new float[mMedianBackgroundTest.channels()];
			for (int ch = 0; (ch < mMedianBackgroundTest.channels()); ch++)
			{
				mHistogram[row][col][ch] = new float[mNumberOfBins];
				mLessThanMedian[row][col][ch] = 0.0;
				for (int bin = 0; (bin < mNumberOfBins); bin++)
				{
					mHistogram[row][col][ch][bin] = (float)0.0;
				}
			}
		}
	}
}

Mat MedianBackgroundTest::GetBackgroundImage()
{
	return mMedianBackgroundTest;
}

void MedianBackgroundTest::UpdateBackground(Mat current_frame)
{
	mTotalAges += mCurrentAge;
	float total_divided_by_2 = mTotalAges / ((float)2.0);
	for (int row = 0; (row < mMedianBackgroundTest.rows); row++)
	{
		for (int col = 0; (col < mMedianBackgroundTest.cols); col++)
		{
			for (int ch = 0; (ch < mMedianBackgroundTest.channels()); ch++)
			{
				int new_value = (mMedianBackgroundTest.channels() == 3) ? current_frame.at<Vec3b>(row, col)[ch] : current_frame.at<uchar>(row, col);
				int median = (mMedianBackgroundTest.channels() == 3) ? mMedianBackgroundTest.at<Vec3b>(row, col)[ch] : mMedianBackgroundTest.at<uchar>(row, col);
				int bin = new_value / mValuesPerBin;
				mHistogram[row][col][ch][bin] += mCurrentAge;
				if (new_value < median)
					mLessThanMedian[row][col][ch] += mCurrentAge;
				int median_bin = median / mValuesPerBin;
				while ((mLessThanMedian[row][col][ch] + mHistogram[row][col][ch][median_bin] < total_divided_by_2) && (median_bin < 255))
				{
					mLessThanMedian[row][col][ch] += mHistogram[row][col][ch][median_bin];
					median_bin++;
				}
				while ((mLessThanMedian[row][col][ch] > total_divided_by_2) && (median_bin > 0))
				{
					median_bin--;
					mLessThanMedian[row][col][ch] -= mHistogram[row][col][ch][median_bin];
				}
				if (mMedianBackgroundTest.channels() == 3)
					mMedianBackgroundTest.at<Vec3b>(row, col)[ch] = median_bin * mValuesPerBin;
				else mMedianBackgroundTest.at<uchar>(row, col) = median_bin * mValuesPerBin;
			}
		}
	}
	mCurrentAge *= mAgingRate;
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
	int starting_frame = 2;
	bool clean_binary_images = false;

	if (video.isOpened())
	{
		Mat current_frame, closed_image, running_average_background;
		Mat selective_running_average_background, temp_selective_running_average_background, selective_running_average_difference;
		Mat selective_running_average_foreground_mask, selective_running_average_foreground_image;
		double running_average_learning_rate = 0.01;
		video.set(cv::CAP_PROP_POS_FRAMES, starting_frame);
		video >> current_frame;
		current_frame.convertTo(running_average_background, CV_32F);
		selective_running_average_background = running_average_background.clone();

		double frame_rate = video.get(cv::CAP_PROP_FPS);
		double time_between_frames = 1000.0 / frame_rate;
		int frame_count = 0;
		while ((!current_frame.empty()) && (frame_count++ < 1000))//1800))
		{
			double duration = static_cast<double>(getTickCount());
			vector<Mat> input_planes(3);
			split(current_frame, input_planes);
				
			// Running Average with selective update
			vector<Mat> selective_running_average_planes(3);
			// Find Foreground mask
			selective_running_average_background.convertTo(temp_selective_running_average_background, CV_8U);
			absdiff(temp_selective_running_average_background, current_frame, selective_running_average_difference);
			split(selective_running_average_difference, selective_running_average_planes);
			// Determine foreground points as any point with an average difference of more than 30 over all channels:
			Mat temp_sum = (selective_running_average_planes[0] / 3 + selective_running_average_planes[1] / 3 + selective_running_average_planes[2] / 3);
			threshold(temp_sum, selective_running_average_foreground_mask, 30, 255, THRESH_BINARY_INV | THRESH_OTSU);
			imshow("Test", selective_running_average_foreground_mask);
			char t = cv::waitKey();
			// Update background
			split(selective_running_average_background, selective_running_average_planes);
			accumulateWeighted(input_planes[0], selective_running_average_planes[0], running_average_learning_rate, selective_running_average_foreground_mask);
			accumulateWeighted(input_planes[1], selective_running_average_planes[1], running_average_learning_rate, selective_running_average_foreground_mask);
			accumulateWeighted(input_planes[2], selective_running_average_planes[2], running_average_learning_rate, selective_running_average_foreground_mask);
			invertImage(selective_running_average_foreground_mask, selective_running_average_foreground_mask);
			accumulateWeighted(input_planes[0], selective_running_average_planes[0], running_average_learning_rate / 3.0, selective_running_average_foreground_mask);
			accumulateWeighted(input_planes[1], selective_running_average_planes[1], running_average_learning_rate / 3.0, selective_running_average_foreground_mask);
			accumulateWeighted(input_planes[2], selective_running_average_planes[2], running_average_learning_rate / 3.0, selective_running_average_foreground_mask);
			merge(selective_running_average_planes, selective_running_average_background);
			if (clean_binary_images)
			{
				Mat structuring_element(3, 3, CV_8U, Scalar(1));
				morphologyEx(selective_running_average_foreground_mask, closed_image, MORPH_CLOSE, structuring_element);
				morphologyEx(closed_image, selective_running_average_foreground_mask, MORPH_OPEN, structuring_element);
			}
			selective_running_average_foreground_image.setTo(Scalar(0, 0, 0));
			current_frame.copyTo(selective_running_average_foreground_image, selective_running_average_foreground_mask);

			duration = static_cast<double>(getTickCount()) - duration;
			duration /= getTickFrequency() / 1000.0;
			int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
			char c = cv::waitKey(delay);

			char frame_str[100];
			sprintf(frame_str, "Frame = %d", frame_count);
			Mat temp_selective_output = JoinImagesHorizontally(current_frame, frame_str, temp_selective_running_average_background, "Selective Running Average Background", 4);
			Mat selective_output = JoinImagesHorizontally(temp_selective_output, "", selective_running_average_foreground_image, "Foreground", 4);
			imshow("Selective Running Average Background Model", selective_output);
			video >> current_frame;
		}
		cv::destroyAllWindows();
	}
}
