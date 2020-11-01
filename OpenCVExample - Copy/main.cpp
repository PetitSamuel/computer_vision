/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

#define REGENTHOUSE_IMAGE_INDEX 0
#define CAMPANILE1_IMAGE_INDEX 1
#define CAMPANILE2_IMAGE_INDEX 2
#define CREST_IMAGE_INDEX 3
#define OLDLIBRARY_IMAGE_INDEX 4
#define WINDOW1_IMAGE_INDEX 5
#define WINDOW2_IMAGE_INDEX 6
#define WINDOW1_LOCATIONS_IMAGE_INDEX 7
#define WINDOW2_LOCATIONS_IMAGE_INDEX 8
#define BIKES_IMAGE_INDEX 9
#define PEOPLE2_IMAGE_INDEX 10
#define ASTRONAUT_IMAGE_INDEX 11
#define PEOPLE1_IMAGE_INDEX 12
#define PEOPLE1_SKIN_MASK_IMAGE_INDEX 13
#define SKIN_IMAGE_INDEX 14
#define CHURCH_IMAGE_INDEX 15
#define FRUIT_IMAGE_INDEX 16
#define COATS_IMAGE_INDEX 17
#define STATIONARY_IMAGE_INDEX 18
#define PETS124_IMAGE_INDEX 19
#define PETS129_IMAGE_INDEX 20
#define PCB_IMAGE_INDEX 21
#define LICENSE_PLATE_IMAGE_INDEX 22
#define BICYCLE_BACKGROUND_IMAGE_INDEX 23
#define BICYCLE_MODEL_IMAGE_INDEX 24
#define NUMBERS_IMAGE_INDEX 25
#define GOOD_ORINGS_IMAGE_INDEX 26
#define BAD_ORINGS_IMAGE_INDEX 27
#define UNKNOWN_ORINGS_IMAGE_INDEX 28

#define SURVEILLANCE_VIDEO_INDEX 0
#define BICYCLES_VIDEO_INDEX 1
#define ABANDONMENT_VIDEO_INDEX 2

#define HAAR_FACE_CASCADE_INDEX 0

int main(int argc, const char** argv)
{
	char* file_location = "Media/";
	char* image_files[] = {
		"TrinityRegentHouse.jpg", //0
		"TrinityCampanile1.jpg",
		"TrinityCampanile3.jpg",
		"TrinityCrest.jpg",
		"TrinityOldLibrary.jpg",
	    "TrinityWindow1.jpg", //5
	    "TrinityWindow2.jpg",
	    "TrinityWindow1Locations.png",
	    "TrinityWindow2Locations.png",
		"TrinityBikes1.jpg",
		"People2.jpg", //10
		"Astronaut2.jpg",
		"People1.jpg",
		"People1SkinMask.jpg",
		"SkinSamples.jpg",
		"Church.jpg", //15
		"FruitStall.jpg",
		"CoatHanger.jpg" ,
		"Stationery.jpg",
		"PETS2000Frame0124.jpg",
		"PETS2000Frame0129.jpg", //20
		"TrinityCampanile3.jpg" ,
	    "LicensePlate1.jpg",
		"BicycleBackgroundImage.jpg",
		"BicycleModel2.jpg",
		"Numbers.jpg", //25
		"GoodORings.jpg",
		"BadORings.jpg",
		"UnknownORings.jpg",
    };

	// Load images
	int number_of_images = sizeof(image_files)/sizeof(image_files[0]);
	Mat* image = new Mat[number_of_images];
	for (int file_no=0; (file_no < number_of_images); file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		image[file_no] = imread(filename, -1);
		if (image[file_no].empty())
		{
			cout << "Could not open " << image[file_no] << endl;
			return -1;
		}
	}



	// Needed for mean shift in histogram demos
	Rect Surveillance_car_position_frame_124(251,164,64,32);
	Rect Bicycles_position_frame_180(242,26,37,60);
	Rect Person_position_frame_100(507,110,67,90);
	// Load video(s)
	char* video_files[] = { 
		"PETS2000_mjpeg.avi",
		"Bicycles_mjpeg.avi",
		"ObjectAbandonmentAndRemoval1_mjpeg.avi",
		"PostboxesWithLines.avi"
	};
	int number_of_videos = sizeof(video_files)/sizeof(video_files[0]);
	VideoCapture* video = new VideoCapture[number_of_videos];
	for (int video_file_no=0; (video_file_no < number_of_videos); video_file_no++)
	{
		string filename(file_location);
		filename.append(video_files[video_file_no]);
		video[video_file_no].open(filename);
		if( !video[video_file_no].isOpened() )
		{
			cout << "Cannot open video file: " << filename << endl;
//			return -1;
		}
	}

	// Load Haar Cascade(s)
	vector<CascadeClassifier> cascades;
	char* cascade_files[] = { 
		"haarcascades/haarcascade_frontalface_alt.xml" };
	int number_of_cascades = sizeof(cascade_files)/sizeof(cascade_files[0]);
	for (int cascade_file_no=0; (cascade_file_no < number_of_cascades); cascade_file_no++)
	{
		CascadeClassifier cascade;
		string filename(file_location);
		filename.append(cascade_files[cascade_file_no]);
		if( !cascade.load( filename ) )
		{
			cout << "Cannot load cascade file: " << filename << endl;
			return -1;
		}
		else cascades.push_back(cascade);
	}

	int line_step = 13;
	Point location( 7, 13 );
	Scalar colour( 0, 0, 255);
	Mat default_image = ComputeDefaultImage( image[CAMPANILE1_IMAGE_INDEX] );
	putText( default_image, "OpenCV demonstration system from:", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step*3/2;
	putText( default_image, "    A PRACTICAL INTRODUCTION TO COMPUTER VISION WITH OPENCV", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "     by Kenneth Dawson-Howe (C) John Wiley & Sons, Inc. 2019", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step*5/2;
	putText( default_image, "Menu choices:", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step*3/2;
	putText( default_image, "1. Images (Sampling+Quantisation, Colour Models, Noise+Smoothing)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "2. Histograms (Histograms, Equalisation, Selection, Back Proj)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText(default_image, "3. Binary Vision (Thresholding, Morphology)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText(default_image, "4. Region Segmentation (Connected Components, k-means, Mean Shift)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText( default_image, "5. Geometric models (Transformation, Interpolation)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "6. Edges (Roberts, Sobel, Laplacian, Colour, Sharpening, Line, Hough)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "7. Features (Features and Feature Matching)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "8. Recognition (Statistics, Templates, Chamfer, Haar and HoG)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "9. Video Processing (Background & Optical Flow, followed by Mean Shift)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText(default_image, "c. Camera Calibration", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText(default_image, "m. My Application", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText( default_image, "X. eXit", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	Mat imageROI;
	imageROI = default_image(cv::Rect(0,0,default_image.cols,245));
	addWeighted(imageROI,2.5,imageROI,0.0,0.0,imageROI);

	int choice;
	do
	{
		imshow("Welcome", default_image);
		choice = cv::waitKey();
		cv::destroyAllWindows();
		switch (choice)
		{
		case 'a':
			VideoDemos(video[3],5,false);
			break;
		case '1':
			ImagesDemos(image[CHURCH_IMAGE_INDEX],image[FRUIT_IMAGE_INDEX],
				image[CREST_IMAGE_INDEX],image[ASTRONAUT_IMAGE_INDEX]);
			break;
		case '2':
			HistogramsDemos(image[CAMPANILE2_IMAGE_INDEX], image[FRUIT_IMAGE_INDEX],
				image[PEOPLE2_IMAGE_INDEX], image[SKIN_IMAGE_INDEX],
				image, number_of_images);
			break;
		case '3':
			BinaryDemos(image[PCB_IMAGE_INDEX], image[STATIONARY_IMAGE_INDEX]);
			break;
		case '4':
			RegionDemos(image[PCB_IMAGE_INDEX], image[COATS_IMAGE_INDEX], image[FRUIT_IMAGE_INDEX]);
			break;
		case '5':
			GeometricDemos(image[LICENSE_PLATE_IMAGE_INDEX],
				image[PETS124_IMAGE_INDEX],image[PETS129_IMAGE_INDEX]);
			break;
		case '6':
			EdgeDemos(image[BIKES_IMAGE_INDEX],image[COATS_IMAGE_INDEX]);
			break;
		case '7':
			FeaturesDemos(image[CHURCH_IMAGE_INDEX],
				image[PETS124_IMAGE_INDEX],image[PETS129_IMAGE_INDEX]);
			TrackFeaturesDemo( video[SURVEILLANCE_VIDEO_INDEX], 120, 159 );
			break;
		case '8':
			RecognitionDemos(image[OLDLIBRARY_IMAGE_INDEX],image[WINDOW1_IMAGE_INDEX],
				image[WINDOW2_IMAGE_INDEX],image[WINDOW1_LOCATIONS_IMAGE_INDEX],
				image[WINDOW2_LOCATIONS_IMAGE_INDEX],video[BICYCLES_VIDEO_INDEX],
				image[BICYCLE_BACKGROUND_IMAGE_INDEX],image[BICYCLE_MODEL_IMAGE_INDEX],
				video[SURVEILLANCE_VIDEO_INDEX],cascades[HAAR_FACE_CASCADE_INDEX],
				image[NUMBERS_IMAGE_INDEX],image[GOOD_ORINGS_IMAGE_INDEX],image[BAD_ORINGS_IMAGE_INDEX],image[UNKNOWN_ORINGS_IMAGE_INDEX]);
			break;
		case '9':
			VideoDemos(video[SURVEILLANCE_VIDEO_INDEX], 120, false);
			// MeanShiftDemo(video[ABANDONMENT_VIDEO_INDEX],Person_position_frame_100,100,230);
			break;
		case 'c':
			{
				string filename(file_location);
				filename.append("default.xml");
				CameraCalibration( filename );
			}
			break;
		case 'm':
			MyApplication();
			break;
		default:
			break;
		}
	} while ((choice != 'x') && (choice != 'X'));
}

