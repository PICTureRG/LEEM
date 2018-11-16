/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright( C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even ifadvised of the possibility of such damage.
//
//M*/
#include "opencv2/opencv.hpp"
#include "precomp.hpp"
#include <iostream>
#include <ctime>
#include <set>
using namespace std;

int distCount = 0;

namespace cv
{
namespace ml
{

const double minEigenValue = DBL_EPSILON;

class CV_EXPORTS EMImpl : public EM //CV_EXPORTS means __declspec(dllexport) means this is an export function
{
  public:
    int nclusters;
    int covMatType;
    TermCriteria termCrit;

    CV_IMPL_PROPERTY_S(TermCriteria, TermCriteria, termCrit)

    void setClustersNumber(int val)
    {
        nclusters = val;
        CV_Assert(nclusters >= 1); //ncluster must be >= 1
    }

    void setClustersNumberToBeConsidered(int val)
    {
        numClstToCnsd = val;
        CV_Assert(numClstToCnsd >= 1); //the number of mixture components to be considered must be >= 1
    }

    int getClustersNumber() const
    {
        return nclusters;
    }

    void setCovarianceMatrixType(int val)
    {
        covMatType = val;
        CV_Assert(covMatType == COV_MAT_SPHERICAL ||
                  covMatType == COV_MAT_DIAGONAL ||
                  covMatType == COV_MAT_GENERIC);
    }
    
    double getMStepTime()
    {
        return mStepTime;
    }

    double getEStepTime()
    {
        return eStepTime;
    }

    int getIterNo()
    {
        return iterNo;
    }

    int getCovarianceMatrixType() const
    {
        return covMatType;
    }

    int getDistCount()
    {
        return distCount;
    }

    vector<Mat> getCovMats()
    {
        return covs;
    }

    Mat getMeans()
    {
        return means;
    }
    
    Mat getWeights()
    {
        return weights;
    }

    EMImpl()
    {
        nclusters = DEFAULT_NCLUSTERS; //DEFAULT_NCLUSTERS = 5
        covMatType = EM::COV_MAT_DIAGONAL;
        termCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, 1e-6); //DEFAULT_MAX_ITERS = 100, EPS = 1e-6
    }

    virtual ~EMImpl() {}

    void clear()
    {
        trainSamples.release();
        trainProbs.release();
        trainLogLikelihoods.release();
        trainLabels.release();

        weights.release();
        means.release();
        covs.clear();

        covsEigenValues.clear();
        invCovsEigenValues.clear();
        covsRotateMats.clear();

        logWeightDivDet.release();

        mDistSqBtwCtrs.release();
        newWeights.release();

        numEachClst.release();
        oldTrainLabels.release();
        oldMeans.release();

        sampleInA.clear();
        sampleInB.clear();
    }

    bool train(const Ptr<TrainData> &data, int)
    {
        Mat samples = data->getTrainSamples(), labels;
        return trainEM(samples, labels, noArray(), noArray());
    }

    bool trainHardEM(InputArray samples, //InputArray is read-only, we can pass Mat,vector to InputArray
                     OutputArray logLikelihoods,
                     OutputArray labels, bool opt)
    {
        eStepTime = 0;
        mStepTime = 0;

        iterNo = 0;
        Mat samplesMat = samples.getMat();
        setTrainData(START_AUTO_STEP, samplesMat, 0, 0, 0, 0); //check the format of data and do necessary data conversion. data are stored in variables in EMImpl class
        bool flag;
        isFirstTime = true;

        if (!opt)
            flag = doTrain_HardEM(START_AUTO_STEP, logLikelihoods, labels);
        else
            flag = doTrain_HardEM_opt(START_AUTO_STEP, logLikelihoods, labels);

        return flag;
    }

    bool trainEM(InputArray samples, //InputArray is read-only, we can pass Mat,vector to InputArray
                 OutputArray logLikelihoods,
                 OutputArray labels,
                 OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
        setTrainData(START_AUTO_STEP, samplesMat, 0, 0, 0, 0); //check the format of data and do necessary data conversion. data are stored in variables in EMImpl class
        return doTrain(START_AUTO_STEP, logLikelihoods, labels, probs);
    }

    bool trainE(InputArray samples,
                InputArray _means0,
                InputArray _covs0,
                InputArray _weights0,
                OutputArray logLikelihoods,
                OutputArray labels,
                OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
        std::vector<Mat> covs0;
        _covs0.getMatVector(covs0);

        Mat means0 = _means0.getMat(), weights0 = _weights0.getMat();

        setTrainData(START_E_STEP, samplesMat, 0, !_means0.empty() ? &means0 : 0,
                     !_covs0.empty() ? &covs0 : 0, !_weights0.empty() ? &weights0 : 0);
        return doTrain(START_E_STEP, logLikelihoods, labels, probs);
    }

    bool trainM(InputArray samples,
                InputArray _probs0,
                OutputArray logLikelihoods,
                OutputArray labels,
                OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
        Mat probs0 = _probs0.getMat();

        setTrainData(START_M_STEP, samplesMat, !_probs0.empty() ? &probs0 : 0, 0, 0, 0);
        return doTrain(START_M_STEP, logLikelihoods, labels, probs);
    }

    float predict(InputArray _inputs, OutputArray _outputs, int) const
    {
        bool needprobs = _outputs.needed();
        Mat samples = _inputs.getMat(), probs, probsrow;
        int ptype = CV_64F;
        float firstres = 0.f;
        int i, nsamples = samples.rows;

        if (needprobs)
        {
            if (_outputs.fixedType())
                ptype = _outputs.type();
            _outputs.create(samples.rows, nclusters, ptype);
            probs = _outputs.getMat();
        }
        else
            nsamples = std::min(nsamples, 1);

        for (i = 0; i < nsamples; i++)
        {
            if (needprobs)
                probsrow = probs.row(i);
            Vec2d res = computeProbabilities(samples.row(i), needprobs ? &probsrow : 0, ptype);
            if (i == 0)
                firstres = (float)res[1];
        }
        return firstres;
    }

    Vec2d predict2(InputArray _sample, OutputArray _probs) const
    {
        int ptype = CV_64F;
        Mat sample = _sample.getMat();
        CV_Assert(isTrained()); //whether we have already trained model

        CV_Assert(!sample.empty());
        if (sample.type() != CV_64FC1)
        {
            Mat tmp;
            sample.convertTo(tmp, CV_64FC1);
            sample = tmp;
        }

        sample = sample.reshape(1, 1); //the first one is for channel, 2nd one is for row, thus change to row vector

        Mat probs;
        if (_probs.needed())
        {
            if (_probs.fixedType())
                ptype = _probs.type();
            _probs.create(1, nclusters, ptype); //create space for _probs then linking to probs
            probs = _probs.getMat();
        }

        return computeProbabilities(sample, !probs.empty() ? &probs : 0, ptype);
    }

    Vec2d predict3(InputArray _sample, OutputArray _probs) const
    {
        int ptype = CV_64F;
        Mat sample = _sample.getMat();
        CV_Assert(isTrained()); //whether we have already trained model

        CV_Assert(!sample.empty());
        if (sample.type() != CV_64FC1)
        {
            Mat tmp;
            sample.convertTo(tmp, CV_64FC1);
            sample = tmp;
        }

        sample = sample.reshape(1, 1); //the first one is for channel, 2nd one is for row, thus change to row vector

        Mat probs;
        if (_probs.needed())
        {
            if (_probs.fixedType())
                ptype = _probs.type();
            _probs.create(1, nclusters, ptype); //create space for _probs then linking to probs
            probs = _probs.getMat();
        }

        return computeProbabilities(sample, !probs.empty() ? &probs : 0, ptype);
    }

    bool isTrained() const
    {
        return !means.empty();
    }

    bool isClassifier() const
    {
        return true;
    }

    int getVarCount() const
    {
        return means.cols;
    }

    String getDefaultName() const
    {
        return "opencv_ml_em";
    }

    static void checkTrainData(int startStep, const Mat &samples,
                               int nclusters, int covMatType, const Mat *probs, const Mat *means,
                               const std::vector<Mat> *covs, const Mat *weights) //the last 4 elements could be NULL
    {
        // Check samples.
        CV_Assert(!samples.empty());
        CV_Assert(samples.channels() == 1);

        int nsamples = samples.rows; //100
        int dim = samples.cols;      //2
        //recover std::cout<<"nsamples:"<<nsamples<<"\ndim"<<dim<<std::endl;

        // Check training params.
        CV_Assert(nclusters > 0);
        CV_Assert(nclusters <= nsamples);         //the number of cluster must be larger than the number of samples
        CV_Assert(startStep == START_AUTO_STEP || //startStep can be any of START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0
                  startStep == START_E_STEP ||
                  startStep == START_M_STEP);
        CV_Assert(covMatType == COV_MAT_GENERIC ||
                  covMatType == COV_MAT_DIAGONAL ||
                  covMatType == COV_MAT_SPHERICAL);

        CV_Assert(!probs ||
                  (!probs->empty() &&
                   probs->rows == nsamples && probs->cols == nclusters &&
                   (probs->type() == CV_32FC1 || probs->type() == CV_64FC1)));

        CV_Assert(!weights ||
                  (!weights->empty() &&
                   (weights->cols == 1 || weights->rows == 1) && static_cast<int>(weights->total()) == nclusters &&
                   (weights->type() == CV_32FC1 || weights->type() == CV_64FC1)));

        CV_Assert(!means ||
                  (!means->empty() &&
                   means->rows == nclusters && means->cols == dim &&
                   means->channels() == 1));

        CV_Assert(!covs ||
                  (!covs->empty() &&
                   static_cast<int>(covs->size()) == nclusters));
        if (covs)
        {
            const Size covSize(dim, dim);             //usage: covSize.width, covSize.height
            for (size_t i = 0; i < covs->size(); i++) //vector<Mat>* covs
            {
                const Mat &m = (*covs)[i]; //mat is the ith cov matrix
                CV_Assert(!m.empty() && m.size() == covSize && (m.channels() == 1));
            }
        }

        if (startStep == START_E_STEP)
        {
            CV_Assert(means);
        }
        else if (startStep == START_M_STEP)
        {
            CV_Assert(probs);
        }
    }

    //if KMeansInit then use CV_32FC1, otherwise, CV_64FC1
    static void preprocessSampleData(const Mat &src, Mat &dst, int dstType, bool isAlwaysClone)
    {
        if (src.type() == dstType && !isAlwaysClone)
            dst = src;
        else
            src.convertTo(dst, dstType);
    }

    static void preprocessProbability(Mat &probs)
    {
        max(probs, 0., probs); //put the maximal one in the third para

        const double uniformProbability = (double)(1. / probs.cols); //probs here is#include "precomp.hpp" acutally probs0 so it has nsamples * nclusters

        for (int y = 0; y < probs.rows; y++)
        {
            Mat sampleProbs = probs.row(y);

            double maxVal = 0;
            minMaxLoc(sampleProbs, 0, &maxVal);
            if (maxVal < FLT_EPSILON) //FLT_EPSILON = 1.192092896e-07F
                sampleProbs.setTo(uniformProbability);
            else
                normalize(sampleProbs, sampleProbs, 1, 0, NORM_L1); //for {a,b,c} what it does is {a/a+b+c, b/a+b+c, c/a+b+c}
        }
    }

    //startStep can be any of START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0
    //probs0 is the prob of Pi,k which is the prob of seeing sample i in model k
    //means0 is the initial mean of each Gaussian
    //covs0 is the initial covariance matrices  of each Gaussian
    //weight0 is initial weight for each Gaussian
    void setTrainData(int startStep, const Mat &samples,
                      const Mat *probs0,
                      const Mat *means0,
                      const std::vector<Mat> *covs0,
                      const Mat *weights0)
    {
        //trainEM call this function using: setTrainData(START_AUTO_STEP, samplesMat, 0, 0, 0, 0);
        //clean data in EMImpl
        clear();

        checkTrainData(startStep, samples, nclusters, covMatType, probs0, means0, covs0, weights0);
        //start with kmeans or start with E step but without initial weights and covariance matrices
        bool isKMeansInit = (startStep == START_AUTO_STEP) || (startStep == START_E_STEP && (covs0 == 0 || weights0 == 0));
        // Set checked data
        //after preprocess, data of samples are in trainSamples
        preprocessSampleData(samples, trainSamples, isKMeansInit ? CV_32FC1 : CV_64FC1, false);

        // set probs
        if (probs0 && startStep == START_M_STEP) //probs0 is the prob of Pi,k which is the prob of seeing sample i in model k
        {
            //if probs0 is not null copy it to trainProbs
            preprocessSampleData(*probs0, trainProbs, CV_64FC1, true);
            //preprocessProbability is to do L1 normalization
            preprocessProbability(trainProbs); //trainProbs is nsamples * nclusters
        }

        // set weights
        if (weights0 && (startStep == START_E_STEP && covs0))
        {
            weights0->convertTo(weights, CV_64FC1);
            weights = weights.reshape(1, 1); //guaranttee it's a row vector
            preprocessProbability(weights);  //L1 normalization
        }

        // set means

        if (means0 && (startStep == START_E_STEP /* || startStep == START_AUTO_STEP*/))
            means0->convertTo(means, isKMeansInit ? CV_32FC1 : CV_64FC1);

        // set covs
        if (covs0 && (startStep == START_E_STEP && weights0))
        {
            covs.resize(nclusters);
            for (size_t i = 0; i < covs0->size(); i++)
                (*covs0)[i].convertTo(covs[i], CV_64FC1);
        }
    }

    void decomposeCovs()
    {
        CV_Assert(!covs.empty());
        covsEigenValues.resize(nclusters);
        if (covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            CV_Assert(!covs[clusterIndex].empty());
            //SVD::MODIFY_A  means it will modify the decomposed matrix during the#include "precomp.hpp" process for opti, but the results are the same
            //SVD::UV means u and vt will keep the complete matrix of orthogonal vectors
            //SVD: A = U*w*V, note V is already tansposed
            SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);

            if (covMatType == COV_MAT_SPHERICAL)
            {
                double maxSingularVal = svd.w.at<double>(0);                                 //use the largest singular value
                covsEigenValues[clusterIndex] = Mat(1, 1, CV_64FC1, Scalar(maxSingularVal)); //1*1 mat
            }
            else if (covMatType == COV_MAT_DIAGONAL)
            {
                //covsEigenValues[clusterIndex] is a col vector
                covsEigenValues[clusterIndex] = covs[clusterIndex].diag().clone(); //Preserve the original order of eigen values.
            }
            else //COV_MAT_GENERIC
            {
                covsEigenValues[clusterIndex] = svd.w; //each matrix in covsEigenValues is a diag mat with singular value
                covsRotateMats[clusterIndex] = svd.u;
            }

            //if there is 0, it will be replaced by minEigenValue = 2.2204460492503131e-016, it avoids denominator to be 0 in the next step
            //covsEigenValues always has singular values
            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
            invCovsEigenValues[clusterIndex] = 1. / covsEigenValues[clusterIndex];
        }
    }

    void clusterTrainSamples()
    {
        int nsamples = trainSamples.rows;

        // Cluster samples, compute/update means

        // Convert samples and means to 32F, because kmeans requires this type.
        Mat trainSamplesFlt, meansFlt;
        if (trainSamples.type() != CV_32FC1)
            trainSamples.convertTo(trainSamplesFlt, CV_32FC1);
        else
            trainSamplesFlt = trainSamples;
        if (!means.empty())
        {
            if (means.type() != CV_32FC1)
                means.convertTo(meansFlt, CV_32FC1);
            else
                meansFlt = means;
        }

        Mat labels;
        kmeans(trainSamplesFlt, nclusters, labels, //labels is to output labels of samples
               TermCriteria(TermCriteria::COUNT, means.empty() ? 10 : 1, 0.5),
               10, KMEANS_PP_CENTERS, meansFlt); //10 is then number of times executed using diff initial labellings, kmeans return the one with best compactness
                                                 //meansFlt has final centers of clusters

        // Convert samples and means back to 64F.
        // KMeans requires 32F
        CV_Assert(meansFlt.type() == CV_32FC1);
        if (trainSamples.type() != CV_64FC1)
        {
            Mat trainSamplesBuffer;
            trainSamplesFlt.convertTo(trainSamplesBuffer, CV_64FC1);
            trainSamples = trainSamplesBuffer;
        }
        meansFlt.convertTo(means, CV_64FC1);

        //we have mean now
        // Compute weights and covs
        weights = Mat(1, nclusters, CV_64FC1, Scalar(0)); //initialize as all 0s
        numEachClst = Mat(1, nclusters, CV_64FC1, Scalar(0));
        covs.resize(nclusters);
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            Mat clusterSamples;
            for (int sampleIndex = 0; sampleIndex < nsamples; sampleIndex++)
            {
                if (labels.at<int>(sampleIndex) == clusterIndex)
                {
                    const Mat sample = trainSamples.row(sampleIndex);
                    clusterSamples.push_back(sample); //clusterSamples has samples belonging to clusterIndex
                    numEachClst.at<double>(clusterIndex)++;
                }
            }
            CV_Assert(!clusterSamples.empty());

            //CV_COVAR_NORMAL means
            //CV_COVAR_ROWS means input are stored as rows of the sample matrix
            //CV_COVAR_USE_AVG means do not calculate mean, useds passed mean
            //CV_COVAR_SCALE means covariance matrix is scaled by 1/nsamples.
            calcCovarMatrix(clusterSamples, covs[clusterIndex], means.row(clusterIndex),
                            CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_USE_AVG + CV_COVAR_SCALE, CV_64FC1);
            //the number of samples belong to cluster divided by the number of samples
            weights.at<double>(clusterIndex) = static_cast<double>(clusterSamples.rows) / static_cast<double>(nsamples);
            //we have mean, cov, weight now
        }
        trainLabels = labels;
        decomposeCovs();
        for(int i = 0; i < nclusters; i++)
            clstOrder.push_back(i);
        //now we have u w v of Cov mat
    }

    void computeLogWeightDivDet()
    {
        CV_Assert(!covsEigenValues.empty());
        Mat logWeights;
        cv::max(weights, DBL_MIN, weights); //DBL_MIN is min positive value of double
        log(weights, logWeights);

        logWeightDivDet.create(1, nclusters, CV_64FC1);
        // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            double logDetCov = 0.;
            const int evalCount = static_cast<int>(covsEigenValues[clusterIndex].total()); //evalCount the number of singular value
            for (int di = 0; di < evalCount; di++)
                logDetCov += std::log(covsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0));
            //even for the generic case, A = U*W*V, |det(A)| = |det(W)|
            logWeightDivDet.at<double>(clusterIndex) = logWeights.at<double>(clusterIndex) - 0.5 * logDetCov;
            //every element in logWeightDivDet is log(Wk/det(Cov))
        }
    }

    bool doTrain_HardEM(int startStep, OutputArray logLikelihoods, OutputArray labels)
    {   
        int dim = trainSamples.cols;
        //cov, mean, weight, prob0 are all null
        if (covs.empty())
        {
            CV_Assert(weights.empty());
            clusterTrainSamples();
        }

        if (!covs.empty() && covsEigenValues.empty())
        {
            CV_Assert(invCovsEigenValues.empty());
            decomposeCovs();
        }

        double trainLogLikelihood, prevTrainLogLikelihood = 0.;
        int maxIters = (termCrit.type & TermCriteria::MAX_ITER) ? termCrit.maxCount : DEFAULT_MAX_ITERS; //if we set max iteration, then we use what we set, if not, use default DEFAULT_MAX_ITERS=100
        double epsilon = (termCrit.type & TermCriteria::EPS) ? termCrit.epsilon : 0.;                    //we set EPS = 0.1

        timespec t_start, t_end;
        for (int iter = 0;; iter++)
        {
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            eStep_HardEM();
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            long timedif = 1000000 *(t_end.tv_sec-t_start.tv_sec)+(t_end.tv_nsec-t_start.tv_nsec)/1000;
            eStepTime += timedif / 1000.0;
            trainLogLikelihood = sum(trainLogLikelihoods)[0];


            if (iter >= maxIters - 1)
                break;

            double trainLogLikelihoodDelta = trainLogLikelihood - prevTrainLogLikelihood;

            if (iter != 0 &&
                (trainLogLikelihoodDelta < -DBL_EPSILON ||
                 trainLogLikelihoodDelta < epsilon * std::fabs(trainLogLikelihood))) //fabs is to get abs value
                break;

            clock_gettime(CLOCK_MONOTONIC, &t_start);
            mStep_HardEM();
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            timedif = 1000000 *(t_end.tv_sec-t_start.tv_sec)+(t_end.tv_nsec-t_start.tv_nsec)/1000;
            mStepTime += timedif / 1000.0;
            prevTrainLogLikelihood = trainLogLikelihood;
            iterNo++;
        }
        //cout<<"iterNo:"<<iterNo<<endl;
        if (trainLogLikelihood <= -DBL_MAX / 10000.)
        {
            clear();
            return false; //training is unsuccessful
        }

        // postprocess covs
        covs.resize(nclusters);

        if (labels.needed())
            trainLabels.copyTo(labels);

        if (logLikelihoods.needed())
            trainLogLikelihoods.copyTo(logLikelihoods);

        trainSamples.release();
        trainLabels.release();
        trainLogLikelihoods.release();

        return true;
    }

    bool doTrain_HardEM_opt(int startStep, OutputArray logLikelihoods, OutputArray labels)
    {
        int dim = trainSamples.cols;

        //cov, mean, weight, prob0 are all null
        if (covs.empty())
        {
            CV_Assert(weights.empty());
            clusterTrainSamples();
        }

        if (!covs.empty() && covsEigenValues.empty())
        {
            CV_Assert(invCovsEigenValues.empty());
            decomposeCovs();
        }

        double trainLogLikelihood, prevTrainLogLikelihood = 0.;
        int maxIters = (termCrit.type & TermCriteria::MAX_ITER) ? termCrit.maxCount : DEFAULT_MAX_ITERS; //if we set max iteration, then we use what we set, if not, use default DEFAULT_MAX_ITERS=100
        double epsilon = (termCrit.type & TermCriteria::EPS) ? termCrit.epsilon : 0.;                    //we set EPS = 0.1

        timespec t_start, t_end;
        for (int iter = 0;; iter++)
        { 
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            eStep_HardEM_opt();
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            long timedif = 1000000 *(t_end.tv_sec-t_start.tv_sec)+(t_end.tv_nsec-t_start.tv_nsec)/1000;
            eStepTime += timedif / 1000.0;
            trainLogLikelihood = sum(trainLogLikelihoods)[0];

            if (iter >= maxIters - 1)
                break;

            double trainLogLikelihoodDelta = trainLogLikelihood - prevTrainLogLikelihood;

            if (iter != 0 &&
                (trainLogLikelihoodDelta < -DBL_EPSILON ||
                 trainLogLikelihoodDelta < epsilon * std::fabs(trainLogLikelihood))) //fabs is to get abs value
                break;
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            if(numClstToCnsd == 1)
                mStep_HardEM_opt();
            else
                mStep_HardEM();
            clock_gettime(CLOCK_MONOTONIC, &t_end);
            timedif = 1000000 *(t_end.tv_sec-t_start.tv_sec)+(t_end.tv_nsec-t_start.tv_nsec)/1000;
            mStepTime += timedif / 1000.0;
            prevTrainLogLikelihood = trainLogLikelihood;
            iterNo++;
        }
        //cout<<"iterNo:"<<iterNo<<endl;
        if (trainLogLikelihood <= -DBL_MAX / 10000.)
        {
            clear();
            return false; //training is unsuccessful
        }

        // postprocess covs
        covs.resize(nclusters);

        if (labels.needed())
            trainLabels.copyTo(labels);

        if (logLikelihoods.needed())
            trainLogLikelihoods.copyTo(logLikelihoods);

        trainSamples.release();
        trainLabels.release();
        trainLogLikelihoods.release();

        return true;
    }

    bool doTrain(int startStep, OutputArray logLikelihoods, OutputArray labels, OutputArray probs)
    {
        int dim = trainSamples.cols;
        // Precompute the empty initial train data in the cases of START_E_STEP and START_AUTO_STEP
        //cov, mean, weight, prob0 are all null
        if (startStep != START_M_STEP)
        {
            if (covs.empty())
            {
                CV_Assert(weights.empty());
                clusterTrainSamples();
            }
        }

        if (!covs.empty() && covsEigenValues.empty())
        {
            CV_Assert(invCovsEigenValues.empty());
            decomposeCovs();
        }

        if (startStep == START_M_STEP)
            mStep();

        double trainLogLikelihood, prevTrainLogLikelihood = 0.;
        int maxIters = (termCrit.type & TermCriteria::MAX_ITER) ? termCrit.maxCount : DEFAULT_MAX_ITERS; //if we set max iteration, then we use what we set, if not, use default DEFAULT_MAX_ITERS=100
        double epsilon = (termCrit.type & TermCriteria::EPS) ? termCrit.epsilon : 0.;                    //we set EPS = 0.1

        for (int iter = 0;; iter++)
        {
            eStep();
            trainLogLikelihood = sum(trainLogLikelihoods)[0];

            if (iter >= maxIters - 1)
                break;

            double trainLogLikelihoodDelta = trainLogLikelihood - prevTrainLogLikelihood;

            if (iter != 0 &&
                (trainLogLikelihoodDelta < -DBL_EPSILON ||
                 trainLogLikelihoodDelta < epsilon * std::fabs(trainLogLikelihood)))
                break;

            mStep();

            prevTrainLogLikelihood = trainLogLikelihood;
        }

        if (trainLogLikelihood <= -DBL_MAX / 10000.)
        {
            clear();
            return false; //training is unsuccessful
        }

        // postprocess covs
        covs.resize(nclusters);
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (covMatType == COV_MAT_SPHERICAL)
            {
                covs[clusterIndex].create(dim, dim, CV_64FC1);
                setIdentity(covs[clusterIndex], Scalar(covsEigenValues[clusterIndex].at<double>(0)));
            }
            else if (covMatType == COV_MAT_DIAGONAL)
            {
                covs[clusterIndex] = Mat::diag(covsEigenValues[clusterIndex]);
            }
        }

        if (labels.needed())
            trainLabels.copyTo(labels);
        if (probs.needed())
            trainProbs.copyTo(probs);
        if (logLikelihoods.needed())
            trainLogLikelihoods.copyTo(logLikelihoods);

        trainSamples.release();
        trainProbs.release();
        trainLabels.release();
        trainLogLikelihoods.release();

        return true;
    }

    Vec2d computeProbabilities(const Mat &sample, Mat *probs, int ptype) const
    {
        //computeProbabilities(trainSamples.row(sampleIndex), &sampleProbs, CV_64F);
        //Thus, sample is one sample, probs is the prob for the sample in diff Gaussian

        // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
        // q = arg(max_k(L_ik))
        // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))  pls see formula (1)
        // see Alex Smola's blog http://blog.smola.org/page/2 for
        // details on the log-sum-exp trick

        int stype = sample.type();
        CV_Assert(!means.empty());
        CV_Assert((stype == CV_32F || stype == CV_64F) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert(sample.size() == Size(means.cols, 1));

        int dim = sample.cols;

        Mat L(1, nclusters, CV_64FC1), centeredSample(1, dim, CV_64F);
        int i, label = 0;
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            const double *mptr = means.ptr<double>(clusterIndex);
            double *dptr = centeredSample.ptr<double>();
            if (stype == CV_32F)
            {
                const float *sptr = sample.ptr<float>();
                for (i = 0; i < dim; i++)
                    dptr[i] = sptr[i] - mptr[i]; //every point substract its mean in each dim
            }

            else
            {
                const double *sptr = sample.ptr<double>();
                for (i = 0; i < dim; i++)
                    dptr[i] = sptr[i] - mptr[i];
            }

            Mat rotatedCenteredSample = covMatType != COV_MAT_GENERIC ? centeredSample : centeredSample * covsRotateMats[clusterIndex];

            double Lval = 0;
            for (int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;

            //now L is the prob of seeing sample i from Gaussian[clusterIndex],  namely firstly choosing Gaussian[clusterIndex], then sample i generated by that Gaussian model
            if (L.at<double>(clusterIndex) > L.at<double>(label)) //update label if we find one with larger prob
                label = clusterIndex;
        }

        double maxLVal = L.at<double>(label); //maxLVal is the max prob
        double expDiffSum = 0;
        for (i = 0; i < L.cols; i++)
        {
            double v = std::exp(L.at<double>(i) - maxLVal);
            L.at<double>(i) = v; //now L is exp(L_ij - L_iq) here i is the sample, j is the Gaussian
            expDiffSum += v;     // sum_j(exp(L_ij - L_iq))
        }

        if (probs)
            L.convertTo(*probs, ptype, 1. / expDiffSum);

        Vec2d res;
        res[0] = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
        res[1] = label;

        return res;
    }

    Vec2d computeProbabilities_HardEM_opt(const Mat &sample, Mat *probs, int ptype, int smp_idx)
    {
        int stype = sample.type();
        CV_Assert(!means.empty());
        CV_Assert((stype == CV_32F || stype == CV_64F) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert(sample.size() == Size(means.cols, 1));

        int dim = sample.cols;

        Mat L(1, nclusters, CV_64FC1), centeredSample(1, dim, CV_64F);
        Mat eDistSqs(1, nclusters, CV_64FC1, -1); //square of Euclidean distance
        int i, label = 0;
        int prevClstIdx = trainLabels.at<int>(smp_idx); //previous cluster index; previous cluster will be calculated first, since most sample will not change their cluster
        
        set< pair<double, int> > clstToCnsd; //clusters to be considered;
        for (int j = 0; j < nclusters; j++)
        {
            int clusterIndex = clstOrder[j];
            //int clusterIndex = j;   
            //if (j == 0)
                //clusterIndex = prevClstIdx;  //first clusterIndex is supposed to be previous cluster, thus switch it with 0
            //else if (j == prevClstIdx)
                //clusterIndex = 0;

            const double *mptr = means.ptr<double>(clusterIndex);
            double *dptr = centeredSample.ptr<double>();
            const double *sptr = sample.ptr<double>();
            for (i = 0; i < dim; i++)
                dptr[i] = sptr[i] - mptr[i];
            
            //centeredSample = sample - means.row(clusterIndex);
            if(clstToCnsd.size() == numClstToCnsd)
            {
                double eDist = norm(centeredSample); //Euclidean distance
                double eDistSq = eDist * eDist;
                eDistSqs.at<double>(clusterIndex) = eDistSq;
                double lBound = eDistSq * invCovsEigenValues[clusterIndex].at<double>(0);
                double uBound = eDistSq * invCovsEigenValues[clusterIndex].at<double>(dim - 1);
                
                bool flag = false; //flag is true means triangle inequality (TI) filter works, calculations for this cluster is filtered out
                if (j != 0)   //for the first cluster, do not use filters, because first distance needs to be calculated
                {
                    if (logWeightDivDet.at<double>(clusterIndex) - 0.5 * lBound <= clstToCnsd.begin()->first && clstToCnsd.size() == numClstToCnsd)
                    //if (logWeightDivDet.at<double>(clusterIndex) - 0.5 * lBound <= L.at<double>(label))
                        continue;
                    //double lb, ub, ctrDist;
                    double lbSq, ubSq, ctrDistSq;
                    for (int k = 0; k < clusterIndex; k++) //get bounds based on previous Euclidean dist multiply singular values in current cluster
                    {
                        if (eDistSqs.at<double>(k) == -1) //since the order is not from 1 to n, so check whether this dist is available
                            continue;

                        lbSq = eDistSqs.at<double>(k) * invCovsEigenValues[clusterIndex].at<double>(0);
                        ubSq = eDistSqs.at<double>(k) * invCovsEigenValues[clusterIndex].at<double>(dim - 1);
                        ctrDistSq = mDistSqBtwCtrs.at<double>(k, clusterIndex); //dist square btw ctrs in current cluster env

                        if (ctrDistSq > ubSq)
                        {
                            lBound = ctrDistSq + ubSq - 2 * sqrt(ctrDistSq * ubSq);
                            if (logWeightDivDet.at<double>(clusterIndex) - 0.5 * lBound <= clstToCnsd.begin()->first && clstToCnsd.size() == numClstToCnsd)
                            //if (logWeightDivDet.at<double>(clusterIndex) - 0.5 * lBound <= L.at<double>(label))
                            {
                                flag = true;
                                break;
                            }
                        }
                        else if (ctrDistSq < lbSq)
                        {
                            lBound = lbSq + ctrDistSq - 2 * sqrt(lbSq * ctrDistSq);
                            if (logWeightDivDet.at<double>(clusterIndex) - 0.5 * lBound <= clstToCnsd.begin()->first && clstToCnsd.size() == numClstToCnsd)
                            //if (logWeightDivDet.at<double>(clusterIndex) - 0.5 * lBound <= L.at<double>(label))
                            {
                                flag = true;
                                break;
                            }
                        }
                    }
                    if (flag)
                        continue;
                }
            }
            //filters do not work, it's necessary to calculate the Mahalanobis distance
            distCount++;
            Mat rotatedCenteredSample = covMatType != COV_MAT_GENERIC ? centeredSample : centeredSample * covsRotateMats[clusterIndex];

            double Lval = 0;

            for (int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;
            if(clstToCnsd.size() < numClstToCnsd || L.at<double>(clusterIndex) > clstToCnsd.begin()->first)
            {
                clstToCnsd.insert(make_pair(L.at<double>(clusterIndex), clusterIndex));
                if(clstToCnsd.size() > numClstToCnsd)
                    clstToCnsd.erase(clstToCnsd.begin());
            }

            //now L is the prob of seeing sample i from Gaussian[clusterIndex],  namely the prob of firstly choosing Gaussian[clusterIndex], then sample i generated by that Gaussian model
            if (j == 0 || L.at<double>(clusterIndex) > L.at<double>(label)) //update label if we find one with larger prob
                label = clusterIndex;
        }
        
        double probSum = 0;
        for(set< pair<double, int> >::iterator it = clstToCnsd.begin(); it != clstToCnsd.end(); it++)
            probSum += weights.at<double>(it->second);
        
        double L_adj = std::log(1.0 / probSum); //since may not consider all cluster, so the total prob may not be 1, thus, we scale considered clusters to make their total prob = 1
        //double maxLVal = L.at<double>(label); //maxLVal is the max prob
        double maxLVal = clstToCnsd.rbegin()->first; //maxLVal is the max prob
        maxLVal += L_adj;
        double expDiffSum = 0;
        
        L = Scalar(0);
        clstOrder.clear();
        vector<bool> isAdded(nclusters, false);//mark whether a cluster is added to clstOrder
        for(set< pair<double, int> >::iterator it = clstToCnsd.begin(); it != clstToCnsd.end(); it++)
        {
            double v = std::exp(it->first + L_adj - maxLVal);
            int index = it->second;
            L.at<double>(index) = v;
            expDiffSum += v;
            clstOrder.push_back(index);
            isAdded[index] = true;
        }
        for(int i1 = 0; i1 < nclusters; i1++)
        {
            if(!isAdded[i1])
                clstOrder.push_back(i1);
        }
        //L = newL;

        /*for (i = 0; i < L.cols; i++)
        {
            double v = std::exp(L.at<double>(i) - maxLVal);
            L.at<double>(i) = v; //now L is exp(L_ij - L_iq) here i is the sample, j is the Gaussian
            expDiffSum += v;     // sum_j(exp(L_ij - L_iq))
        }*/

        if(probs)
            L.convertTo(*probs, ptype, 1. / expDiffSum);
        Vec2d res;
        res[0] = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
        res[1] = label;
        return res;
    }

    Vec2d computeProbabilities_HardEM(const Mat &sample, Mat* probs, int ptype)
    {
        int stype = sample.type();
        CV_Assert(!means.empty());
        CV_Assert((stype == CV_32F || stype == CV_64F) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert(sample.size() == Size(means.cols, 1));

        int dim = sample.cols;
        Mat L(1, nclusters, CV_64FC1), centeredSample(1, dim, CV_64F);
        int i, label = 0;
        set< pair<double, int> > clstToCnsd; //clusters to be considered;
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            const double *mptr = means.ptr<double>(clusterIndex);
            double *dptr = centeredSample.ptr<double>();
            if (stype == CV_32F)
            {
                const float *sptr = sample.ptr<float>();
                for (i = 0; i < dim; i++)
                    dptr[i] = sptr[i] - mptr[i]; //every point substract its mean in each dim
            }

            else
            {
                const double *sptr = sample.ptr<double>();
                for (i = 0; i < dim; i++)
                    dptr[i] = sptr[i] - mptr[i];
            }
            distCount++;
            Mat rotatedCenteredSample = covMatType != COV_MAT_GENERIC ? centeredSample : centeredSample * covsRotateMats[clusterIndex];

            double Lval = 0;
            for (int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;
            //now L is the prob of seeing sample i from Gaussian[clusterIndex],  namely firstly choosing Gaussian[clusterIndex], then sample i generated by that Gaussian model
            clstToCnsd.insert(make_pair(L.at<double>(clusterIndex), clusterIndex));
            if (L.at<double>(clusterIndex) > L.at<double>(label)) //update label if we find one with larger prob
                label = clusterIndex;
        }
        
        double probSum = 0;
        int counter = 0;
        vector<double> tmpsum;
        for(set< pair<double, int> >::iterator it = --clstToCnsd.end(); it != clstToCnsd.begin(); it--)
        {
            probSum += weights.at<double>(it->second);
            if(++counter == numClstToCnsd)
                break;
        }

        double L_adj = std::log(1.0 / probSum); //since may not consider all cluster, so the total prob may not be 1, thus, we scale considered clusters to make their total prob = 1
            
        double maxLVal = clstToCnsd.rbegin()->first; //maxLVal is the max prob
        maxLVal += L_adj;
        double expDiffSum = 0;

        L = Scalar(0);
        counter = 0;
        for(set< pair<double, int> >::iterator it = --clstToCnsd.end(); it != clstToCnsd.begin(); it--)
        {
            double v = std::exp(it->first + L_adj - maxLVal);
            int index = it->second;
            L.at<double>(index) = v;
            expDiffSum += v;
            if(++counter == numClstToCnsd)
                break;
        }

        if(probs)
            L.convertTo(*probs, ptype, 1. / expDiffSum);

        Vec2d res;
        res[0] = std::log(expDiffSum) + maxLVal - 0.5 * dim * CV_LOG2PI;
        res[1] = label;
        return res;
    }

    void compCtrDistBtwItr() 
    {
        mDistSqBtwCtrsBtwItr.create(1, nclusters, CV_64FC1); //Mahalonobis distance square between same cluster's centers of two consecutive iterations
        int dim = trainSamples.cols;
        Mat vecBtwCtrsBtwItr(1, dim, CV_64F); //vector btw same cluster's centers of two consecutive iterations
        for (int i = 0; i < nclusters; i++)
        {
            const double *iptr = means.ptr<double>(i);
            double *vptr = vecBtwCtrsBtwItr.ptr<double>();
            
            const double *jptr = oldMeans.ptr<double>(i);
            for (int k = 0; k < dim; k++)
                vptr[k] = iptr[k] - jptr[k];

            Mat rotVecBtwCtrsBtwItr = covMatType != COV_MAT_GENERIC ? //rotVecBtwMeans is the vector after rotation
            vecBtwCtrsBtwItr: vecBtwCtrsBtwItr * covsRotateMats[i];

            distCount++;
            double distSq = 0;
            for (int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[i].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                double val = rotVecBtwCtrsBtwItr.at<double>(di);
                distSq += w * val * val;
            }
            mDistSqBtwCtrsBtwItr.at<double>(i) = distSq;
        }
    }

    void compCtrDist()
    {
        mDistSqBtwCtrs.create(nclusters, nclusters, CV_64FC1); //Mahalonobis distance square between cluster centers
        int dim = trainSamples.cols;
        Mat vecBtwCtrs(1, dim, CV_64F); //vector btw two centers
        for (int i = 0; i < nclusters - 1; i++)
        {
            const double *iptr = means.ptr<double>(i);
            double *vptr = vecBtwCtrs.ptr<double>();
            for (int j = i + 1; j < nclusters; j++)
            {
                const double *jptr = means.ptr<double>(j);
                for (int k = 0; k < dim; k++)
                    vptr[k] = iptr[k] - jptr[k];

                Mat rotVecBtwCtrs = covMatType != COV_MAT_GENERIC ? //rotVecBtwMeans is the vector after rotation
                vecBtwCtrs: vecBtwCtrs * covsRotateMats[j];

                double distSq = 0;
                for (int di = 0; di < dim; di++)
                {
                    double w = invCovsEigenValues[j].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                    double val = rotVecBtwCtrs.at<double>(di);
                    distSq += w * val * val;
                }
                mDistSqBtwCtrs.at<double>(i, j) = distSq;
            }
        }
    }

    void eStep_HardEM()
    {
        int dim = trainSamples.cols;
        sampleInA.clear();
        sampleInB.clear();
        sampleInA.resize(nclusters, vector<bool>(trainSamples.rows, false));
        sampleInB.resize(nclusters, vector<bool>(trainSamples.rows, false));

        trainProbs.create(trainSamples.rows, nclusters, CV_64FC1);
        trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);
        computeLogWeightDivDet();

        CV_DbgAssert(trainSamples.type() == CV_64FC1); //check the condition only in Debug mode
        CV_DbgAssert(means.type() == CV_64FC1);
        
        for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            Mat sampleProbs = trainProbs.row(sampleIndex);  //sampleProbs are probs for one sample in diff Gaussians
            Vec2d res = computeProbabilities_HardEM(trainSamples.row(sampleIndex), &sampleProbs, CV_64F); //res[0] is the log likelihood of sample
            
            trainLogLikelihoods.at<double>(sampleIndex) = res[0];
            trainLabels.at<int>(sampleIndex) = static_cast<int>(res[1]);
        }

    }

    void eStep_HardEM_opt()
    {
        timespec t_start, t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        int dim = trainSamples.cols;
        sampleInA.clear();
        sampleInB.clear();
        sampleInA.resize(nclusters, vector<bool>(trainSamples.rows, false));
        sampleInB.resize(nclusters, vector<bool>(trainSamples.rows, false));

        trainProbs.create(trainSamples.rows, nclusters, CV_64FC1);
        trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);
        
        computeLogWeightDivDet();
        compCtrDist();

        CV_DbgAssert(trainSamples.type() == CV_64FC1); //check the condition only in Debug mode
        CV_DbgAssert(means.type() == CV_64FC1);

        trainLabels.copyTo(oldTrainLabels);

        for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            Mat sampleProbs = trainProbs.row(sampleIndex);                                         //sampleProbs are probs for one sample in diff Gaussians
            Vec2d res = computeProbabilities_HardEM_opt(trainSamples.row(sampleIndex), &sampleProbs, CV_64F, sampleIndex); //res[0] is the log likelihood of sample

            trainLogLikelihoods.at<double>(sampleIndex) = res[0];
            int newLabel = static_cast<int>(res[1]);
            int oldLabel = trainLabels.at<int>(sampleIndex);
            
            if (oldLabel != newLabel) //there are pts leaving the old cluster and joining the new cluster
            {
                sampleInA[oldLabel][sampleIndex] = true;
                sampleInB[newLabel][sampleIndex] = true;
            }

            trainLabels.at<int>(sampleIndex) = newLabel;
        }
    }

    Mat precisionAdj(Mat cov) //truncate precision to avoid precision error caused by double
    {
        Mat cov_32FC1, intermediateCov, mask, newCov;
        cov.convertTo(cov_32FC1, CV_32FC1);
        cov_32FC1.convertTo(intermediateCov, CV_64FC1);
        inRange(intermediateCov, -FLT_EPSILON, FLT_EPSILON, mask); //value smaller than FLF_EPSILON is considered as precision error, ignored
        bitwise_not(mask, mask);
        intermediateCov.copyTo(newCov, mask);
        return newCov;
    }

    void mStep_HardEM_opt()
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int dim = trainSamples.cols;
        reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        means.copyTo(oldMeans);
        means = Scalar(0); //initialized as 0

        const double minPosWeight = trainSamples.rows * DBL_EPSILON;
        double minWeight = DBL_MAX;
        int minWeightClusterIndex = -1;

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            if (weights.at<double>(clusterIndex) < minWeight)
            {
                minWeight = weights.at<double>(clusterIndex);
                minWeightClusterIndex = clusterIndex;
            }
            Mat clusterMean = means.row(clusterIndex);
            for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                if (trainLabels.at<int>(sampleIndex) == clusterIndex)
                {
                    Mat sample = trainSamples.row(sampleIndex);
                    clusterMean += sample;
                }
            }
            clusterMean /= weights.at<double>(clusterIndex);
        }

        Mat deltaMeans = means - oldMeans;

        // Update covsEigenValues and invCovsEigenValues
        covs.resize(nclusters);
        covsEigenValues.resize(nclusters);
        if (covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);
        if (!isFirstTime) //original method is used for the first iteration, then optimization is applied
        {
            for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
            {
                int m = weights.at<double>(clusterIndex);  //number of samples in the cluster after current iteration
                int n = static_cast<int>(numEachClst.at<double>(clusterIndex));  //number of samples in the cluster before current iteration
                Mat clusterCov = covs[clusterIndex];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        clusterCov.at<double>(i, j) = n / (long double)m * clusterCov.at<double>(i, j) + n / (long double)m * deltaMeans.at<double>(clusterIndex, i) * deltaMeans.at<double>(clusterIndex, j);
            }
            for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                int oldLabel = oldTrainLabels.at<int>(sampleIndex);
                int newLabel = trainLabels.at<int>(sampleIndex);
                
                Mat centeredSample;
                if (oldLabel != newLabel)
                {
                    int m = weights.at<double>(oldLabel);
                    centeredSample = trainSamples.row(sampleIndex) - means.row(oldLabel);
                    covs[oldLabel] -= 1.0 / m * centeredSample.t() * centeredSample;
                    m = weights.at<double>(newLabel);
                    centeredSample = trainSamples.row(sampleIndex) - means.row(newLabel);
                    covs[newLabel] += 1.0 / m * centeredSample.t() * centeredSample;
                }
            }
        }
        
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            covsEigenValues[clusterIndex].create(1, dim, CV_64FC1);
            Mat clusterCov = covs[clusterIndex];

            int m = weights.at<double>(clusterIndex);  //number of samples in the cluster after current iteration
            int n = static_cast<int>(numEachClst.at<double>(clusterIndex));  //number of samples in the cluster before current iteration

            Mat centeredSample;
            if (isFirstTime) //original method is used for the first iteration, then optimization is applied
            {
                clusterCov = Scalar(0); //initialize as 0
                for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
                {
                    if (trainLabels.at<int>(sampleIndex) == clusterIndex)
                    {
                        centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);
                        clusterCov += centeredSample.t() * centeredSample;
                    }
                }
                clusterCov /= weights.at<double>(clusterIndex);
            }

            // Update covsRotateMats for COV_MAT_GENERIC only
            if (covMatType == COV_MAT_GENERIC)
            {
                SVD svd(clusterCov, SVD::MODIFY_A + SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
            // update invCovsEigenValues
            invCovsEigenValues[clusterIndex] = 1. / covsEigenValues[clusterIndex];
        }
        isFirstTime = false;

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
            {
                cout << "replacement happens !" << endl;
                Mat clusterMean = means.row(clusterIndex);
                means.row(minWeightClusterIndex).copyTo(clusterMean);
                covs[minWeightClusterIndex].copyTo(covs[clusterIndex]);
                covsEigenValues[minWeightClusterIndex].copyTo(covsEigenValues[clusterIndex]);
                covsRotateMats[minWeightClusterIndex].copyTo(covsRotateMats[clusterIndex]);
                invCovsEigenValues[minWeightClusterIndex].copyTo(invCovsEigenValues[clusterIndex]);
            }
        }

        // Normalize weights
        weights.copyTo(numEachClst);
        weights /= trainSamples.rows;
    }

    void mStep_HardEM()
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int dim = trainSamples.cols;
        reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        means = Scalar(0); //initialized as 0

        const double minPosWeight = trainSamples.rows * DBL_EPSILON;
        double minWeight = DBL_MAX;
        int minWeightClusterIndex = -1;

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            if (weights.at<double>(clusterIndex) < minWeight)
            {
                minWeight = weights.at<double>(clusterIndex);
                minWeightClusterIndex = clusterIndex;
            }
            Mat clusterMean = means.row(clusterIndex);
            for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                if(trainProbs.at<double>(sampleIndex, clusterIndex) > 0)
                    clusterMean += trainProbs.at<double>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex);
            }
            clusterMean /= weights.at<double>(clusterIndex);
        }

        // Update covsEigenValues and invCovsEigenValues
        covs.resize(nclusters);
        covsEigenValues.resize(nclusters);
        covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);
        
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            covsEigenValues[clusterIndex].create(1, dim, CV_64FC1);

            Mat clusterCov = covs[clusterIndex];
            clusterCov = Scalar(0);

            Mat centeredSample;
            for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);

                if (covMatType == COV_MAT_GENERIC && trainProbs.at<double>(sampleIndex, clusterIndex) > 0)
                    clusterCov += trainProbs.at<double>(sampleIndex, clusterIndex) * centeredSample.t() * centeredSample;
            }

            clusterCov /= weights.at<double>(clusterIndex);

            // Update covsRotateMats for COV_MAT_GENERIC only
            if (covMatType == COV_MAT_GENERIC)
            {
                SVD svd(clusterCov, SVD::MODIFY_A + SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
            // update invCovsEigenValues
            invCovsEigenValues[clusterIndex] = 1. / covsEigenValues[clusterIndex];
        }

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
            {
                cout << "replacement happens !" << endl;
                Mat clusterMean = means.row(clusterIndex);
                means.row(minWeightClusterIndex).copyTo(clusterMean);
                covs[minWeightClusterIndex].copyTo(covs[clusterIndex]);
                covsEigenValues[minWeightClusterIndex].copyTo(covsEigenValues[clusterIndex]);
                covsRotateMats[minWeightClusterIndex].copyTo(covsRotateMats[clusterIndex]);
                invCovsEigenValues[minWeightClusterIndex].copyTo(invCovsEigenValues[clusterIndex]);
            }
        }

        // Normalize weights
        weights /= trainSamples.rows;
    }

    void eStep()
    {
        // Compute probs_ik from means_k, covs_k and weights_k.

        trainProbs.create(trainSamples.rows, nclusters, CV_64FC1);
        trainLabels.create(trainSamples.rows, 1, CV_32SC1); //col vector
        trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1);

        computeLogWeightDivDet();

        CV_DbgAssert(trainSamples.type() == CV_64FC1); //check the condition only in Debug mode
        CV_DbgAssert(means.type() == CV_64FC1);

        for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            Mat sampleProbs = trainProbs.row(sampleIndex);                                         //sampleProbs are probs for one sample in diff Gaussians
            Vec2d res = computeProbabilities(trainSamples.row(sampleIndex), &sampleProbs, CV_64F); //res[0] is the log likelihood of sample
            trainLogLikelihoods.at<double>(sampleIndex) = res[0];
            trainLabels.at<int>(sampleIndex) = static_cast<int>(res[1]);
        }
    }

    void mStep()
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int dim = trainSamples.cols;

        // Update weights
        // not normalized first
        reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        // Update means
        means.create(nclusters, dim, CV_64FC1);
        means = Scalar(0); //initialized as 0

        const double minPosWeight = trainSamples.rows * DBL_EPSILON;
        double minWeight = DBL_MAX;
        int minWeightClusterIndex = -1;
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            if (weights.at<double>(clusterIndex) < minWeight)
            {
                minWeight = weights.at<double>(clusterIndex);
                minWeightClusterIndex = clusterIndex;
            }

            Mat clusterMean = means.row(clusterIndex);
            for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
                clusterMean += trainProbs.at<double>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex);
            clusterMean /= weights.at<double>(clusterIndex);
        }

        // Update covsEigenValues and invCovsEigenValues
        covs.resize(nclusters);
        covsEigenValues.resize(nclusters);
        if (covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters);
        invCovsEigenValues.resize(nclusters);
        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

            if (covMatType != COV_MAT_SPHERICAL)
                covsEigenValues[clusterIndex].create(1, dim, CV_64FC1);
            else
                covsEigenValues[clusterIndex].create(1, 1, CV_64FC1);

            if (covMatType == COV_MAT_GENERIC)
                covs[clusterIndex].create(dim, dim, CV_64FC1);

            Mat clusterCov = covMatType != COV_MAT_GENERIC ? covsEigenValues[clusterIndex] : covs[clusterIndex];

            clusterCov = Scalar(0); //initialize as 0

            Mat centeredSample;
            for (int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);

                if (covMatType == COV_MAT_GENERIC)
                    clusterCov += trainProbs.at<double>(sampleIndex, clusterIndex) * centeredSample.t() * centeredSample;
                else
                {
                    double p = trainProbs.at<double>(sampleIndex, clusterIndex);
                    for (int di = 0; di < dim; di++)
                    {
                        double val = centeredSample.at<double>(di);
                        clusterCov.at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0) += p * val * val;
                    }
                }
            }

            if (covMatType == COV_MAT_SPHERICAL)
                clusterCov /= dim;

            clusterCov /= weights.at<double>(clusterIndex);

            // Update covsRotateMats for COV_MAT_GENERIC only
            if (covMatType == COV_MAT_GENERIC)
            {
                SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);

            // update invCovsEigenValues
            invCovsEigenValues[clusterIndex] = 1. / covsEigenValues[clusterIndex];
        }

        for (int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if (weights.at<double>(clusterIndex) <= minPosWeight)
            {
                Mat clusterMean = means.row(clusterIndex);
                means.row(minWeightClusterIndex).copyTo(clusterMean);
                covs[minWeightClusterIndex].copyTo(covs[clusterIndex]);
                covsEigenValues[minWeightClusterIndex].copyTo(covsEigenValues[clusterIndex]);
                if (covMatType == COV_MAT_GENERIC)
                    covsRotateMats[minWeightClusterIndex].copyTo(covsRotateMats[clusterIndex]);
                invCovsEigenValues[minWeightClusterIndex].copyTo(invCovsEigenValues[clusterIndex]);
            }
        }

        // Normalize weights
        weights /= trainSamples.rows;
    }

    void write_params(FileStorage &fs) const
    {
        fs << "nclusters" << nclusters;
        fs << "cov_mat_type" << (covMatType == COV_MAT_SPHERICAL ? String("spherical") : covMatType == COV_MAT_DIAGONAL ? String("diagonal") : covMatType == COV_MAT_GENERIC ? String("generic") : format("unknown_%d", covMatType));
        writeTermCrit(fs, termCrit);
    }

    void write(FileStorage &fs) const
    {
        writeFormat(fs);
        fs << "training_params"
           << "{";
        write_params(fs);
        fs << "}";
        fs << "weights" << weights;
        fs << "means" << means;

        size_t i, n = covs.size();

        fs << "covs"
           << "[";
        for (i = 0; i < n; i++)
            fs << covs[i];
        fs << "]";
    }

    void read_params(const FileNode &fn)
    {
        nclusters = (int)fn["nclusters"];
        String s = (String)fn["cov_mat_type"];
        covMatType = s == "spherical" ? COV_MAT_SPHERICAL : s == "diagonal" ? COV_MAT_DIAGONAL : s == "generic" ? COV_MAT_GENERIC : -1;
        CV_Assert(covMatType >= 0);
        termCrit = readTermCrit(fn);
    }

    void read(const FileNode &fn)
    {
        clear();
        read_params(fn["training_params"]);

        fn["weights"] >> weights;
        fn["means"] >> means;

        FileNode cfn = fn["covs"];
        FileNodeIterator cfn_it = cfn.begin();
        int i, n = (int)cfn.size();
        covs.resize(n);

        for (i = 0; i < n; i++, ++cfn_it)
            (*cfn_it) >> covs[i];

        decomposeCovs();
        computeLogWeightDivDet();
    }

    Mat getWeights() const { return weights; }
    Mat getMeans() const { return means; }
    void getCovs(std::vector<Mat> &_covs) const
    {
        _covs.resize(covs.size());
        std::copy(covs.begin(), covs.end(), _covs.begin());
    }

    // all inner matrices have type CV_64FC1
    Mat trainSamples;
    Mat trainProbs;
    Mat trainLogLikelihoods;
    Mat trainLabels;

    Mat weights;
    Mat means;
    std::vector<Mat> covs;

    std::vector<Mat> covsEigenValues;
    std::vector<Mat> covsRotateMats;
    std::vector<Mat> invCovsEigenValues;
    Mat logWeightDivDet;

    Mat mDistSqBtwCtrs; //square of mahalanobis dist btw ctrs, mDistSqBtwCtrs.at<double>(i,j) using cluster j's cov and  mean
    Mat mDistSqBtwCtrsBtwItr; //square of mahalanobis dist btw ctrs of the same cluster of two consecutive iteration

    Mat newWeights;         //store new weights in E step
    Mat numEachClst;

    vector< vector<bool> > sampleInA, sampleInB; //record whether a sample is in A(leaving) and B(joining) of a cluster
    bool isFirstTime;

    double eStepTime;
    double mStepTime;

    Mat oldMeans; 
    int iterNo; //iteration Number starts from 1
    Mat oldTrainLabels;   

    int numClstToCnsd; //the number of mixture components to be considered
    vector<int> clstOrder;
};

Ptr<EM> EM::create()
{
    return makePtr<EMImpl>(); //construct a EMImpl class and return its pointer
}

Ptr<EM> EM::load(const String &filepath, const String &nodeName)
{
    return Algorithm::load<EM>(filepath, nodeName);
}
}
} // namespace cv

/* End of file. */
