#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>
#include <limits>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

#define MILLION 1000000
//Using EM implement clustering and prediction
int main(int argc, char** argv)
{
	bool opt;
	if(strcmp(argv[4], "true") == 0)
	    opt = true;
	else
	    opt = false;

    const int T = atoi(argv[1]); //number of category

	vector<Mat> samples(T);
	Mat completeSamples; //have all samples
	Mat completeSampleCategories;

	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::loadFromCSV(argv[5], 0);
	Mat part1 = train_data->getSamples();
	part1.convertTo(completeSamples, CV_64FC1);
	Mat part2 = train_data->getResponses();
	part2.convertTo(completeSampleCategories, CV_64FC1);
	for(int i = 0; i < completeSamples.rows; i++)
	{
		int categoryIndex = static_cast<int>(completeSampleCategories.at<double>(i,0));
		samples[categoryIndex].push_back(completeSamples.row(i));
	}

	Mat testData;
	cv::Ptr<cv::ml::TrainData> test_data = cv::ml::TrainData::loadFromCSV(argv[6], 0);
	Mat testDataRaw = test_data->getSamples();
    testDataRaw.convertTo(testData, CV_64FC1);

    const int M = atoi(argv[2]);    //number of mixture components
	const int K = atoi(argv[3]);    //number of mixture components to be considered
    
    // train
    vector< pair<int, double> > res(testData.rows, pair<int, double>(0, -DBL_MAX));
	//timespec t_start, t_end;
	//clock_gettime(CLOCK_MONOTONIC, &t_start);

	double total_timedif = 0;
	int total_iterNo = 0;
	double total_mStepTime = 0;
	double total_eStepTime = 0;
	double total_distCount = 0;
	for(int i = 0; i < T; i++)
	{
        Mat labels;  //to be labeled, indicating the Gaussian it belongs to 
		Ptr<EM> em_model = EM::create();
		em_model->setClustersNumber(M);
		em_model->setClustersNumberToBeConsidered(K);
        em_model->setCovarianceMatrixType(EM::COV_MAT_GENERIC);
        em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.00001));
        timespec t_start, t_end;
	    clock_gettime(CLOCK_MONOTONIC, &t_start);
	    //em_model->trainEM(samples[i], noArray(), labels, noArray());
		em_model->trainHardEM(samples[i], noArray(), labels, opt);
		clock_gettime(CLOCK_MONOTONIC, &t_end);
	    total_timedif += MILLION*(t_end.tv_sec-t_start.tv_sec)+(t_end.tv_nsec-t_start.tv_nsec)/1000;

        //cout<<em_model->getDistCount()<<" "<<em_model->getIterNo()<<" "<<(double)em_model->getDistCount() / (double)em_model->getIterNo()<<endl;
        total_iterNo += em_model->getIterNo();
		total_eStepTime += em_model->getEStepTime();
		total_mStepTime += em_model->getMStepTime();
		total_distCount += em_model->getDistCount();
		
        //cout<<em_model->getDistCount()<<endl;
    
	    vector<Mat> covs = em_model->getCovMats();
		//for(int tt = 0; tt < covs.size(); tt++)
		    //cout<<covs[tt]<<endl;
        Mat weights = em_model->getWeights();
		Mat means = em_model->getMeans();

		for(int j = 0; j < testData.rows; j++)
	    {
		    double prob = cvRound(em_model->predict2(testData.row(j), noArray())[0]);
		    if(prob > res[j].second)
			{
				res[j].second = prob;
				res[j].first = i;
			}
	    }
		
	}

	//clock_gettime(CLOCK_MONOTONIC, &t_end);
	//long timedif = MILLION*(t_end.tv_sec-t_start.tv_sec)+(t_end.tv_nsec-t_start.tv_nsec)/1000;
    //printf("it took %lf ms\n", timedif / 1000.0);

	printf("it took %lf ms\n", total_timedif / 1000.0);
	printf("avg iterTime %lf ms\n", total_timedif / 1000.0 / total_iterNo);
	printf("avg estep iterTime %lf ms\n", total_eStepTime / (double)total_iterNo);
    printf("avg mstep iterTime %lf ms\n", total_mStepTime / (double)total_iterNo);
	printf("avg distCount per iter %lf\n", total_distCount / (double)total_iterNo);

	cout<<total_distCount<<endl;
    //for(int i = 0; i < testData.rows; i++)
	//    cout<<res[i].first<<" ";
	//cout<<endl;
	
    return 0;
}
