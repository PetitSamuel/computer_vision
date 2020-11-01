/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"

class Histogram
{
protected:
	Mat mImage;
	int mNumberChannels;
	int* mChannelNumbers;
	int* mNumberBins;
	float mChannelRange[2];
public:
	Histogram( Mat image, int number_of_bins )
	{
		mImage = image;
		mNumberChannels = mImage.channels();
		mChannelNumbers = new int[mNumberChannels];
		mNumberBins = new int[mNumberChannels];
		mChannelRange[0] = 0.0;
		mChannelRange[1] = 255.0;
		for (int count=0; count<mNumberChannels; count++)
		{
			mChannelNumbers[count] = count;
			mNumberBins[count] = number_of_bins;
		}
		//ComputeHistogram();
	}
	virtual void ComputeHistogram()=0;
	virtual void NormaliseHistogram()=0;
	static void Draw1DHistogram( MatND histograms[], int number_of_histograms, Mat& display_image )
	{
		int number_of_bins = histograms[0].size[0];
		double max_value=0, min_value=0;
		double channel_max_value=0, channel_min_value=0;
		for (int channel=0; (channel < number_of_histograms); channel++)
		{
			minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
			max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
			min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
		}
		float scaling_factor = ((float)256.0)/((float)number_of_bins);
		
		Mat histogram_image((int)(((float)number_of_bins)*scaling_factor)+1,(int)(((float)number_of_bins)*scaling_factor)+1,CV_8UC3,Scalar(255,255,255));
		display_image = histogram_image;
		line(histogram_image,Point(0,0),Point(0,histogram_image.rows-1),Scalar(0,0,0));
		line(histogram_image,Point(histogram_image.cols-1,histogram_image.rows-1),Point(0,histogram_image.rows-1),Scalar(0,0,0));
		int highest_point = static_cast<int>(0.9*((float)number_of_bins)*scaling_factor);
		for (int channel=0; (channel < number_of_histograms); channel++)
		{
			int last_height;
			for( int h = 0; h < number_of_bins; h++ )
			{
				float value = histograms[channel].at<float>(h);
				int height = static_cast<int>(value*highest_point/max_value);
				int where = (int)(((float)h)*scaling_factor);
				if (h > 0)
					line(histogram_image,Point((int)(((float)(h-1))*scaling_factor)+1,(int)(((float)number_of_bins)*scaling_factor)-last_height),
								         Point((int)(((float)h)*scaling_factor)+1,(int)(((float)number_of_bins)*scaling_factor)-height),
							             Scalar(channel==0?255:0,channel==1?255:0,channel==2?255:0));
				last_height = height;
			}
		}
	}
};
class OneDHistogram : public Histogram
{
private:
	MatND mHistogram[3];
public:
	OneDHistogram( Mat image, int number_of_bins ) :
	  Histogram( image, number_of_bins )
	{
		ComputeHistogram( );
	}
	void ComputeHistogram( )
	{
		vector<Mat> image_planes(mNumberChannels);
		split(mImage, image_planes);
		for (int channel=0; (channel < mNumberChannels); channel++)
		{
			const float* channel_ranges = mChannelRange;
			int *mch = {0};
			calcHist(&(image_planes[channel]), 1, mChannelNumbers, Mat(), mHistogram[channel], 1 , mNumberBins, &channel_ranges);
		}
	}
	void SmoothHistogram( )
	{
		for (int channel=0; (channel < mNumberChannels); channel++)
		{
			MatND temp_histogram = mHistogram[channel].clone();
			for(int i = 1; i < mHistogram[channel].rows - 1; ++i)
			{
				mHistogram[channel].at<float>(i) = (temp_histogram.at<float>(i-1) + temp_histogram.at<float>(i) + temp_histogram.at<float>(i+1)) / 3;
			}
		}
	}
	MatND getHistogram(int index)
	{
		return mHistogram[index];
	}
	void NormaliseHistogram()
	{
		for (int channel=0; (channel < mNumberChannels); channel++)
		{
			normalize(mHistogram[channel],mHistogram[channel],1.0);
		}
	}
	Mat BackProject( Mat& image )
	{
		Mat& result = image.clone();
		if (mNumberChannels == 1)
		{
			const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
			for (int channel=0; (channel < mNumberChannels); channel++)
			{
				calcBackProject(&image,1,mChannelNumbers,*mHistogram,result,channel_ranges,255.0);
			}
		}
		else
		{
		}
		return result;
	}
	void Draw( Mat& display_image )
	{
		Draw1DHistogram( mHistogram, mNumberChannels, display_image );
	}
};


class ColourHistogram : public Histogram
{
private:
	MatND mHistogram;
public:
	ColourHistogram(Mat all_images[], int number_of_images, int number_of_bins) :
		Histogram(all_images[0], number_of_bins)
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		for (int index=0; index<number_of_images; index++)
			calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges, true,true);
	}
	ColourHistogram(Mat image, int number_of_bins) :
		Histogram(image, number_of_bins)
	{
		ComputeHistogram();
	}
	void ComputeHistogram()
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges);
	}
	void NormaliseHistogram()
	{
		normalize(mHistogram,mHistogram,1.0);
	}
	Mat BackProject( Mat& image )
	{
		Mat& result = image.clone();
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcBackProject(&image,1,mChannelNumbers,mHistogram,result,channel_ranges,255.0);
		return result;
	}
	MatND getHistogram()
	{
		return mHistogram;
	}
};


class HueHistogram : public Histogram
{
private:
	MatND mHistogram;
	int mMinimumSaturation, mMinimumValue, mMaximumValue;
#define DEFAULT_MIN_SATURATION 25
#define DEFAULT_MIN_VALUE 25
#define DEFAULT_MAX_VALUE 230
public:
	HueHistogram( Mat image, int number_of_bins, int min_saturation=DEFAULT_MIN_SATURATION, int min_value=DEFAULT_MIN_VALUE, int max_value=DEFAULT_MAX_VALUE ) :
	  Histogram( image, number_of_bins )
	{
		mMinimumSaturation = min_saturation;
		mMinimumValue = min_value;
		mMaximumValue = max_value;
		mChannelRange[1] = 180.0;
		ComputeHistogram();
	}
	void ComputeHistogram()
	{
		Mat hls_image, hue_image, mask_image;
		cvtColor(mImage, hls_image, COLOR_BGR2HLS);

		vector<Mat> hls_planes(3);
		split(hls_image, hls_planes);
		Mat saturation_minus_luminance;
		cvtColor(mImage, saturation_minus_luminance, COLOR_BGR2GRAY);
		Mat saturation = hls_planes[2];
		Mat luminance = hls_planes[1];
		for (int row = 0; row < saturation_minus_luminance.rows; row++)
			for (int col = 0; col < saturation_minus_luminance.cols; col++)
			{
				int s = ((int)saturation.at<uchar>(row, col));
				int l = ((int)luminance.at<uchar>(row, col));
				int s_l = (((int)saturation.at<uchar>(row, col)) -(((int)luminance.at<uchar>(row, col)) / 2));
				if (s_l < 0)
					saturation_minus_luminance.at<uchar>(row, col) = 0;
				else saturation_minus_luminance.at<uchar>(row, col) = s_l;
//				saturation_minus_luminance.at<uchar>(row, col) = (((int)saturation.at<uchar>(row, col)) - ((int)luminance.at<uchar>(row, col))) > 0 ?
//					(((int)saturation.at<uchar>(row, col)) - ((int)luminance.at<uchar>(row, col))) : 0;
			}
		Mat temp_image;
		threshold(saturation_minus_luminance, mask_image, 20, 255, THRESH_BINARY); // | THRESH_OTSU);
		//inRange( hls_image, Scalar( 0, mMinimumSaturation, mMinimumValue ), Scalar( 180, 256, mMaximumValue ), mask_image );
		cvtColor(mask_image, temp_image, COLOR_GRAY2BGR);
		//imshow("Original image", mImage);
		//imshow("Mask image", temp_image);
		//char c = cv::waitKey();
		int channels[]={0,0};
		hue_image.create( mImage.size(), mImage.depth());
		mixChannels( &hls_image, 1, &hue_image, 1, channels, 1 );
		const float* channel_ranges = mChannelRange;
		calcHist( &hue_image,1,0,mask_image,mHistogram,1,mNumberBins,&channel_ranges);
	}
	void NormaliseHistogram()
	{
		normalize(mHistogram,mHistogram,0,255,cv::NORM_MINMAX);
	}
	Mat BackProject( Mat& image )
	{
		Mat hls_image, hue_image, mask_image;
		cvtColor(image, hls_image, COLOR_BGR2HLS);
				vector<Mat> hls_planes(3);
		split(hls_image, hls_planes);
		Mat saturation_minus_luminance;
		cvtColor(image, saturation_minus_luminance, COLOR_BGR2GRAY);
		Mat& saturation = hls_planes[2];
		Mat& luminance = hls_planes[1];
		for (int row = 0; row < saturation_minus_luminance.rows; row++)
			for (int col = 0; col < saturation_minus_luminance.cols; col++)
			{
				int s = ((int)saturation.at<uchar>(row, col));
				int l = ((int)luminance.at<uchar>(row, col));
				int s_l = (((int)saturation.at<uchar>(row, col)) - (((int)luminance.at<uchar>(row, col)) / 2));
				if (s_l < 0)
					saturation_minus_luminance.at<uchar>(row, col) = 0;
				else saturation_minus_luminance.at<uchar>(row, col) = s_l;
			}
		Mat temp_image;
		threshold(saturation_minus_luminance, mask_image, 20, 255, THRESH_BINARY); // | THRESH_OTSU);
																				   //inRange( hls_image, Scalar( 0, mMinimumSaturation, mMinimumValue ), Scalar( 180, 256, mMaximumValue ), mask_image );
		//cvtColor(mask_image, temp_image, COLOR_GRAY2BGR);
		//imshow("Query mask", temp_image);
		Mat result;
		//Mat hls_image, hue_image, mask_image;
		//cvtColor(image, hls_image, COLOR_BGR2HLS);
		int channels[] = { 0,0 };
		hue_image.create(image.size(), image.depth());
		mixChannels(&hls_image, 1, &hue_image, 1, channels, 1);
		const float* channel_ranges = mChannelRange;
		calcBackProject(&hue_image,1,mChannelNumbers,mHistogram,result,&channel_ranges,255.0);
		Mat masked_result_image = result & mask_image;
		return masked_result_image.clone();
	}
	MatND getHistogram()
	{
		return mHistogram;
	}
	void Draw( Mat& display_image )
	{
		Draw1DHistogram( &mHistogram, 1, display_image );
	}
};

Mat BackProjection(Mat& query_image, Mat all_images[], int number_of_images)
{
	Mat& result = query_image.clone();
	ColourHistogram histogram3D(all_images, number_of_images, 4);
	histogram3D.NormaliseHistogram();
	Mat back_projection_probabilities = histogram3D.BackProject(query_image);
	//back_projection_probabilities = StretchImage(back_projection_probabilities);
	Mat back_projection_probabilities_display;
	cvtColor(back_projection_probabilities, back_projection_probabilities_display, COLOR_GRAY2BGR);
	Mat output1 = JoinImagesHorizontally(query_image, "Original Image", back_projection_probabilities_display, "Probabilities", 4);
	imshow("Back Projection", output1);
	return back_projection_probabilities_display;
}

Mat BackProjection(Mat& query_image, Mat sample_image)
{
	/*
	MatND hist;
	int histSize = 8;
	float hue_range[] = { 0, 180 };
	const float* ranges = { hue_range }; 

	Mat hsv, hue;
	cvtColor(sample_image, hsv, CV_BGR2HSV);
	/// Use only the Hue value
	hue.create(hsv.size(), hsv.depth());
	int ch[] = { 0, 0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	/// Get the Histogram and normalize it
	calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	cvtColor(query_image, hsv, CV_BGR2HSV);
	/// Use only the Hue value
	hue.create(hsv.size(), hsv.depth());
//	int ch[] = { 0, 0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	/// Get Backprojection
	MatND backproj;
	calcBackProject(&hue, 1, 0, hist, backproj, &ranges, 1, true);

	/// Draw the backproj
	imshow("BackProj", backproj);
	char c = cv::waitKey();
	*/
	Mat& result = query_image.clone();
	Mat histogram_display;
	HueHistogram histogram3D(sample_image, 16);
	histogram3D.Draw(histogram_display);
	histogram3D.NormaliseHistogram();
	Mat back_projection_probabilities = histogram3D.BackProject(query_image);
	back_projection_probabilities = StretchImage(back_projection_probabilities);
	Mat back_projection_probabilities_display;
	cvtColor(back_projection_probabilities, back_projection_probabilities_display, COLOR_GRAY2BGR);
	Mat output1 = JoinImagesHorizontally(query_image, "Original Image", back_projection_probabilities_display, "Probabilities", 4);
	//imshow("Hue Histogram", histogram_display);
	//imshow("Back Projection", output1);
	return back_projection_probabilities_display;
/*
	Mat& result = query_image.clone();
	ColourHistogram histogram3D(sample_image, 4);
	histogram3D.NormaliseHistogram();
	Mat back_projection_probabilities = histogram3D.BackProject(query_image);
	back_projection_probabilities = StretchImage(back_projection_probabilities);
	Mat back_projection_probabilities_display;
	cvtColor(back_projection_probabilities, back_projection_probabilities_display, COLOR_GRAY2BGR);
	Mat output1 = JoinImagesHorizontally(query_image, "Original Image", back_projection_probabilities_display, "Probabilities", 4);
	imshow("Back Projection", output1);
	return back_projection_probabilities_display;
	*/
}

void HistogramsDemos( Mat& dark_image, Mat& fruit_image, Mat& people_image, Mat& skin_image, Mat all_images[], int number_of_images )
{
	// Just so that tests can be done using a grayscale image...
	Mat gray_fruit_image;
	cvtColor(fruit_image, gray_fruit_image, COLOR_BGR2GRAY);
	Timestamper* timer = new Timestamper();

	// Greyscale and Colour (RGB) histograms
	Mat gray_histogram_display_image;
	MatND gray_histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 64;
	calcHist(&gray_fruit_image, 1, channel_numbers, Mat(), gray_histogram, 1, &number_bins, &channel_ranges);
	OneDHistogram::Draw1DHistogram(&gray_histogram, 1, gray_histogram_display_image);
	Mat colour_display_image;
	MatND* colour_histogram = new MatND[ fruit_image.channels() ];
	vector<Mat> colour_channels( fruit_image.channels() );
	split( fruit_image, colour_channels );
	for (int chan=0; chan < fruit_image.channels(); chan++)
		calcHist( &(colour_channels[chan]), 1, channel_numbers, Mat(),
		     colour_histogram[chan], 1, &number_bins, &channel_ranges );
	OneDHistogram::Draw1DHistogram( colour_histogram, fruit_image.channels(), colour_display_image );
	Mat gray_fruit_image_display;
	cvtColor(gray_fruit_image, gray_fruit_image_display, COLOR_GRAY2BGR);
	Mat output1 = JoinImagesHorizontally(gray_fruit_image_display,"Grey scale image",gray_histogram_display_image,"Greyscale histogram",4);
	Mat output2 = JoinImagesHorizontally(fruit_image,"Colour image",colour_display_image,"RGB Histograms",4);
	Mat output3 = JoinImagesHorizontally(output1,"",output2,"",4);
	imshow( "Histograms", output3 );

	// This can also be done using the provided OneDHistogram class:
	//Mat histogram_image;
	//OneDHistogram histogram(fruit_image,64);
	//histogram.Draw(histogram_image);
	//imshow("Histogram", histogram_image);
	char c = cv::waitKey();
    cv::destroyAllWindows();

	// Equalisation
	timer->reset();
	std::vector<cv::Mat> input_planes(3);
	Mat processed_image,original_image;
	resize(dark_image,original_image,Size(dark_image.cols/2,dark_image.rows/2));
	Mat hls_image;
	cvtColor(original_image, hls_image, COLOR_BGR2HLS);
	split(hls_image,input_planes);
	timer->recordTime("Split planes");
	Mat input_luminance_histogram_image;
	OneDHistogram input_luminance_histogram(input_planes[1],256);
	input_luminance_histogram.Draw(input_luminance_histogram_image);
	timer->ignoreTimeSinceLastRecorded();
	equalizeHist(input_planes[1],input_planes[1]);
	timer->recordTime("Equalise");
	Mat output_luminance_histogram_image;
	OneDHistogram output_luminance_histogram(input_planes[1],256);
	output_luminance_histogram.Draw(output_luminance_histogram_image);
	merge(input_planes, hls_image);
	cvtColor(hls_image, processed_image, COLOR_HLS2BGR);
	timer->recordTime("Merge output");
	output1 = JoinImagesHorizontally(original_image,"Original image",input_luminance_histogram_image,"Original Luminance Histogram",4);
	output2 = JoinImagesHorizontally(processed_image,"Equalised image",output_luminance_histogram_image,"Equalised Luminance Histogram",4);
	output3 = JoinImagesHorizontally(output1,"",output2,"",4);
	imshow( "Image (luminance) Equalisation", output3 );
	c = cv::waitKey();
    cv::destroyAllWindows();

	// Image selection based on histogram comparison
	timer->reset();
	int index_of_reference_image = 0;
	Mat& reference_image = all_images[index_of_reference_image];
	cvtColor(all_images[index_of_reference_image], hls_image, COLOR_BGR2HLS);
	ColourHistogram reference_histogram(hls_image,4);
	reference_histogram.NormaliseHistogram();
	double best_matching_score = 0.0;
	Mat* best_match = NULL;
	int display_image_count = 0;
	for (int image_index=0; (image_index<number_of_images); image_index++)
	{
		if (image_index != index_of_reference_image)
		{
			cvtColor(all_images[image_index], hls_image, COLOR_BGR2HLS);
			ColourHistogram comparison_histogram(hls_image,4);

			comparison_histogram.NormaliseHistogram();
			double matching_score = compareHist(reference_histogram.getHistogram(),comparison_histogram.getHistogram(),cv::HISTCMP_CORREL);//cv::HISTCMP_CHISQR);//cv::HISTCMP_INTERSECT);//cv::HISTCMP_BHATTACHARYYA);//
			char output[100];
			sprintf(output,"%.4f",matching_score);
			//sprintf(output,"H%.2f L%.2f S%.2f",matching_score,matching_score2,matching_score3);
			//matching_score = (matching_score+matching_score3)/2;
			char image_name[50];
			sprintf(image_name,"Image %d",image_index);
			Scalar colour( 0, 0, 255 );
			Point location( 7, 13 );
			Mat temp_image = all_images[image_index].clone();
			resize( all_images[image_index].clone(), temp_image, Size( all_images[image_index].cols * (200) / all_images[image_index].rows , 200 ) );
			putText( temp_image, output, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
			if ((matching_score > best_matching_score) || (matching_score > 0.78))
			{
				if (display_image_count++ == 0)
					output1 = temp_image.clone();
				else
				{
					output1 = JoinImagesHorizontally(output1,"",temp_image,"",4);
				}
			}
			if (matching_score > best_matching_score)
			{
				best_matching_score = matching_score;
				best_match = &(all_images[image_index]);
			}
		}
	}
	Mat temp_image, temp_image2;
	resize( reference_image.clone(), temp_image, Size( reference_image.cols * (200) / reference_image.rows , 200 ) );
	if (best_match != NULL)
		resize( best_match->clone(), temp_image2, Size( best_match->cols * (200) / best_match->rows , 200 ) );
	output2 = JoinImagesHorizontally(temp_image,"Reference image",temp_image2,"Best match",4);
	output3 = JoinImagesVertically(output1,"",output2,"",4);
	imshow( "Image Selection", output3 );
	c = cv::waitKey();
    cv::destroyAllWindows();

	// Colour selection - back-projection
	timer->reset();
	cvtColor(skin_image, hls_image, COLOR_BGR2HLS);
	ColourHistogram histogram3D(hls_image,8);
	histogram3D.NormaliseHistogram();
	cvtColor(people_image, hls_image, COLOR_BGR2HLS);
	Mat back_projection_probabilities = histogram3D.BackProject(hls_image);
	back_projection_probabilities = StretchImage( back_projection_probabilities );
	Mat back_projection_probabilities_display;
	cvtColor(back_projection_probabilities, back_projection_probabilities_display, COLOR_GRAY2BGR);
	output1 = JoinImagesHorizontally(people_image,"Original Image",skin_image,"Skin Samples",4);
	output2 = JoinImagesHorizontally(output1,"",back_projection_probabilities_display,"Skin Back Projection",4);
	imshow( "Back Projection", output2 );

	c = cv::waitKey();
    cv::destroyAllWindows();
}
