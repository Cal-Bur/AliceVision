// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/image/all.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/ProgressDisplay.hpp>
#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/utils/filesIO.hpp>
#include <aliceVision/config.hpp>
#include <aliceVision/sfmDataIO/viewIO.hpp>

#include <boost/program_options.hpp>

#include <functional>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <filesystem>
#include <vector>
#include <set>
#include <iterator>
#include <iomanip>
#include <fstream>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 1

using namespace aliceVision;
using namespace aliceVision::camera;
using namespace aliceVision::geometry;
using namespace aliceVision::image;
using namespace aliceVision::sfmData;
using namespace aliceVision::sfmDataIO;

namespace po = boost::program_options;
namespace fs = std::filesystem;

template<class ImageT, class MaskFuncT>
void process(const std::string& dstColorImage,
             const camera::IntrinsicBase & outputIntrinsic,
             const camera::IntrinsicBase & sourceIntrinsic,
             const oiio::ParamValueList& metadata,
             const std::string& srcImage,
             bool evCorrection,
             float exposureCompensation,
             MaskFuncT&& maskFunc)
{
    ImageT image, image_ud;
    readImage(srcImage, image, image::EImageColorSpace::LINEAR);

    // exposure correction
    if (evCorrection)
    {
        for (int pix = 0; pix < image.width() * image.height(); ++pix)
        {
            image(pix)[0] *= exposureCompensation;
            image(pix)[1] *= exposureCompensation;
            image(pix)[2] *= exposureCompensation;
        }
    }

    // mask
    maskFunc(image);
    
    // undistort the image and save it
    using Pix = typename ImageT::Tpixel;
    Pix pixZero(Pix::Zero());
    undistortImage(image, sourceIntrinsic, outputIntrinsic, image_ud, pixZero);

    //Write the result
    writeImage(dstColorImage, image_ud, image::ImageWriteOptions(), metadata);
}

template<typename T/*, typename = std::enable_if_t<std::is_integral_v<T>>*/>
std::string to_string_with_zero_padding(const T& value, std::size_t total_length)
{
    auto str = std::to_string(value);
    if (str.length() < total_length)
        str.insert(str.front() == '-' ? 1 : 0, total_length - str.length(), '0');
    return str;
}

bool prepareDenseScene(const SfMData& sfmData,
                       SfMData & outputSfmData,
                       const std::vector<std::string>& imagesFolders,
                       const std::vector<std::string>& masksFolders,
                       const std::string& maskExtension,
                       int beginIndex,
                       int endIndex,
                       const std::string& outFolder,
                       image::EImageFileType outputFileType,
                       double fakeFov,
                       bool saveMetadata,
                       bool saveMatricesFiles,
                       bool evCorrection)
{
    // defined view Ids
    std::set<IndexT> viewIds = outputSfmData.getValidViews();

    if ((outputFileType != image::EImageFileType::EXR) && saveMetadata)
    {
        ALICEVISION_LOG_WARNING("Cannot save informations in images metadata.\n"
                                "Choose '.exr' file type if you want AliceVision custom metadata");
    }

    // export data
    auto progressDisplay = system::createConsoleProgressDisplay(viewIds.size(), std::cout, "Exporting Scene Undistorted Images\n");

    // for exposure correction
    const double medianCameraExposure = sfmData.getMedianCameraExposureSetting().getExposure();
    ALICEVISION_LOG_INFO("Median Camera Exposure: " << medianCameraExposure << ", Median EV: " << std::log2(1.0 / medianCameraExposure));

#pragma omp parallel for num_threads(3)
    for (int i = 0; i < viewIds.size(); ++i)
    {
        auto itView = viewIds.begin();
        std::advance(itView, i);

        const IndexT viewId = *itView;
        const View & view = sfmData.getView(viewId);

        const auto & originalIntrinsic = sfmData.getIntrinsic(view.getIntrinsicId());
        const auto & intrinsic = outputSfmData.getIntrinsic(view.getIntrinsicId());

        // we have a valid view with a corresponding camera & pose
        const std::string baseFilename = to_string_with_zero_padding(view.getFrameId(), 5);
        

        // get metadata from source image to be sure we get all metadata. 
        // We don't use the metadatas from the Views inside the SfMData to avoid type
        // conversion problems with string maps.
        std::string srcImage = view.getImage().getImagePath();
        oiio::ParamValueList metadata = image::readImageMetadata(srcImage);

        // get camera pose / projection
        const Pose3 pose = sfmData.getPose(view).getTransform();
        const auto & camPinHole = dynamic_cast<const camera::Pinhole &>(intrinsic);

        Mat34 P = camPinHole.getProjectiveEquivalent(pose);

        // get camera intrinsics matrices
        const Mat3 K = camPinHole.K();
        const Mat3& R = pose.rotation();
        const Vec3& t = pose.translation();

        if (saveMatricesFiles)
        {
            std::ofstream fileP((fs::path(outFolder) / (baseFilename + "_P.txt")).string());
            fileP << std::setprecision(10) << P(0, 0) << " " << P(0, 1) << " " << P(0, 2) << " " << P(0, 3) << "\n"
                    << P(1, 0) << " " << P(1, 1) << " " << P(1, 2) << " " << P(1, 3) << "\n"
                    << P(2, 0) << " " << P(2, 1) << " " << P(2, 2) << " " << P(2, 3) << "\n";
            fileP.close();

            std::ofstream fileKRt((fs::path(outFolder) / (baseFilename + "_KRt.txt")).string());
            fileKRt << std::setprecision(10) << K(0, 0) << " " << K(0, 1) << " " << K(0, 2) << "\n"
                    << K(1, 0) << " " << K(1, 1) << " " << K(1, 2) << "\n"
                    << K(2, 0) << " " << K(2, 1) << " " << K(2, 2) << "\n"
                    << "\n"
                    << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << "\n"
                    << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << "\n"
                    << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << "\n"
                    << "\n"
                    << t(0) << " " << t(1) << " " << t(2) << "\n";
            fileKRt.close();
        }

        if (saveMetadata)
        {
            // convert to 44 matix
            Mat4 projectionMatrix;
            projectionMatrix << P(0, 0), P(0, 1), P(0, 2), P(0, 3), P(1, 0), P(1, 1), P(1, 2), P(1, 3), P(2, 0), P(2, 1), P(2, 2), P(2, 3), 0, 0,
                0, 1;

            // convert matrices to rowMajor
            std::vector<double> vP(projectionMatrix.size());
            std::vector<double> vK(K.size());
            std::vector<double> vR(R.size());

            typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;
            Eigen::Map<RowMatrixXd>(vP.data(), projectionMatrix.rows(), projectionMatrix.cols()) = projectionMatrix;
            Eigen::Map<RowMatrixXd>(vK.data(), K.rows(), K.cols()) = K;
            Eigen::Map<RowMatrixXd>(vR.data(), R.rows(), R.cols()) = R;

            // add metadata
            metadata.push_back(oiio::ParamValue("AliceVision:downscale", 1));
            metadata.push_back(oiio::ParamValue("AliceVision:P", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX44), 1, vP.data()));
            metadata.push_back(oiio::ParamValue("AliceVision:K", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX33), 1, vK.data()));
            metadata.push_back(oiio::ParamValue("AliceVision:R", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX33), 1, vR.data()));
            metadata.push_back(oiio::ParamValue("AliceVision:t", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::VEC3), 1, t.data()));
        }

        // export undistort image
        {
            if (!imagesFolders.empty())
            {
                std::vector<std::string> paths = sfmDataIO::viewPathsFromFolders(view, imagesFolders);

                // if path was not found
                if (paths.empty())
                {
                    throw std::runtime_error("Cannot find view '" + std::to_string(view.getViewId()) + "' image file in given folder(s)");
                }
                else if (paths.size() > 1)
                {
                    throw std::runtime_error("Ambiguous case: Multiple source image files found in given folder(s) for the view '" +
                                             std::to_string(view.getViewId()) + "'.");
                }

                srcImage = paths[0];
            }

            const std::string dstColorImage =
              (fs::path(outFolder) / (baseFilename + "." + image::EImageFileType_enumToString(outputFileType))).string();

            // add exposure values to images metadata
            const double cameraExposure = view.getImage().getCameraExposureSetting().getExposure();
            const double ev = std::log2(1.0 / cameraExposure);
            const float exposureCompensation = float(medianCameraExposure / cameraExposure);
            metadata.push_back(oiio::ParamValue("AliceVision:EV", float(ev)));
            metadata.push_back(oiio::ParamValue("AliceVision:EVComp", exposureCompensation));

            if (evCorrection)
            {
                ALICEVISION_LOG_INFO("image " << viewId << ", exposure: " << cameraExposure << ", Ev " << ev
                                              << " Ev compensation: " + std::to_string(exposureCompensation));
            }

            image::Image<unsigned char> mask;

            std::function<void(Image<RGBAfColor>&)> function = [](Image<RGBAfColor>& image) {};
            
            
            if (tryLoadMask(&mask, masksFolders, viewId, srcImage, maskExtension))
            {
                function = [&mask](Image<RGBAfColor>& image) {
                                if (image.width() * image.height() != mask.width() * mask.height())
                                {
                                    ALICEVISION_LOG_WARNING("Invalid image mask size: mask is ignored.");
                                    return;
                                }

                                for (int pix = 0; pix < image.width() * image.height(); ++pix)
                                {
                                    const bool masked = (mask(pix) == 0);
                                    image(pix).a() = masked ? 0.f : 1.f;
                                }
                            };
            }

            process<Image<RGBAfColor>>(dstColorImage, intrinsic, originalIntrinsic, metadata, srcImage, evCorrection, exposureCompensation, function);
        }

        ++progressDisplay;
    }

    return true;
}

/**
 * @brief create a new SfmData where all the non pinhole stuff are removed
 * For example, distortion/undistortion are removed, and the images are undistorted
 * @param sfmData the original sfmData
 * @param outputSfmData the result sfmData
 * @param fakeFov if one intrinsic is non pinhole, what is the required fov for the "fake" camera
 * @param baseImagePath the path to the stored images
 * @param fileType the output images file type
 * @return true if everything worked
*/
bool convertSfmData(const sfmData::SfMData & sfmData, sfmData::SfMData & outputSfmData, double fakeFov, const std::string & baseImagePath, const EImageFileType & fileType)
{
    outputSfmData = sfmData::SfMData(sfmData);

    outputSfmData.getIntrinsics().clear();

    // Loop over all input intrinsics
    for (const auto & [intrinsicId, intrinsicPtr] : sfmData.getIntrinsics())
    {
        const auto & originalIntrinsic = *intrinsicPtr;
        bool isPinhole = camera::isPinhole(originalIntrinsic.getType());

        //BY default, create a fake camera with a given fov
        double hw = double(originalIntrinsic.w()) * 0.5;
        double fx = hw / std::tan(fakeFov * 0.5);
        double fy = fx;
        double cx = 0.0;
        double cy = 0.0;

        if (isPinhole)
        {
            //IF pinhone, recreate without distortion
            const auto & pinhole = dynamic_cast<const camera::Pinhole &>(originalIntrinsic);
            fx = pinhole.getScale().x();
            fy = pinhole.getScale().y();
            cx = pinhole.getOffset().x();
            cy = pinhole.getOffset().y();
        }


        std::shared_ptr<camera::IntrinsicBase> fakecam = camera::createPinhole(camera::DISTORTION_NONE, 
                                                camera::UNDISTORTION_NONE, 
                                                originalIntrinsic.w(), originalIntrinsic.h(),
                                                fx, fy, cx, cy
                                                );

        outputSfmData.getIntrinsics().insert({intrinsicId, fakecam});
    }

    outputSfmData.getViews().clear();
    for (auto & [idView, pView] : sfmData.getViews())
    {
        if (!sfmData.isPoseAndIntrinsicDefined(*pView))
        {
            continue;
        }

        const std::string extName = image::EImageFileType_enumToString(fileType);
        const std::string baseFilename = to_string_with_zero_padding(pView->getFrameId(), 5);
        const std::string path = (fs::path(baseImagePath) / (baseFilename + "." + extName)).string();

        sfmData::View::sptr pOutView(pView->clone());
        pOutView->getImage().setImagePath(path);
        outputSfmData.getViews().insert({idView, pOutView});
    }

    for (auto & [idLandmark, landmark] : outputSfmData.getLandmarks())
    {
        //Copy observations and erase
        const auto observationsCopy = landmark.getObservations();
        landmark.getObservations().clear();

        for (const auto & [idView, obs] : observationsCopy)
        {
            //Ignore non reconstructed views
            const auto & view = sfmData.getView(idView);
            if (!sfmData.isPoseAndIntrinsicDefined(view))
            {
                continue;
            }

            //Undistort observation
            IndexT intrinsicId = view.getIntrinsicId();
            const auto & inputIntrinsic = sfmData.getIntrinsic(intrinsicId);
            const auto & outputIntrinsic = outputSfmData.getIntrinsic(intrinsicId);

            const Vec3 intermediate = inputIntrinsic.backProjectUnit(obs.getCoordinates());
            if (intermediate.z() < 1e-2)
            {
                continue;
            }

            const Vec2 undistorted = outputIntrinsic.project(intermediate.homogeneous(), true);
            if (undistorted.x() < 0 || undistorted.y() < 0)
            {
                continue;
            }

            if (undistorted.x() >= outputIntrinsic.w() || undistorted.y() >= outputIntrinsic.h())
            {
                continue;
            }

            sfmData::Observation outputObservation = obs;
            outputObservation.setCoordinates(undistorted);

            landmark.getObservations()[idView] = outputObservation;
        }
    }

    return true;
}

int aliceVision_main(int argc, char* argv[])
{
    // command-line parameters

    std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
    std::string sfmDataFilename;
    std::string sfmDataOutputFilename;
    std::string outFolder;
    std::string outImageFileTypeName = image::EImageFileType_enumToString(image::EImageFileType::EXR);
    std::vector<std::string> imagesFolders;
    std::vector<std::string> masksFolders;
    std::string maskExtension = "png";
    int rangeStart = -1;
    int rangeSize = 1;
    bool saveMetadata = true;
    bool saveMatricesTxtFiles = false;
    bool evCorrection = false;
    double fakeFov = 90.0;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
         "SfMData file.")
        ("outputSfmData,s", po::value<std::string>(&sfmDataOutputFilename)->required(),
         "SfMData Output File.")
        ("output,o", po::value<std::string>(&outFolder)->required(),
         "Output folder.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("imagesFolders",  po::value<std::vector<std::string>>(&imagesFolders)->multitoken(),
         "Use images from specific folder(s) instead of those specify in the SfMData file.\n"
         "Filename should be the same or the image UID.")
        ("masksFolders", po::value<std::vector<std::string>>(&masksFolders)->multitoken(),
         "Use masks from specific folder(s).\n"
         "Filename should be the same or the image UID.")
        ("maskExtension", po::value<std::string>(&maskExtension)->default_value(maskExtension),
         "File extension of the masks to use.")
        ("outputFileType", po::value<std::string>(&outImageFileTypeName)->default_value(outImageFileTypeName),
         image::EImageFileType_informations().c_str())
        ("saveMetadata", po::value<bool>(&saveMetadata)->default_value(saveMetadata),
         "Save projections and intrinsics information in images metadata.")
        ("saveMatricesTxtFiles", po::value<bool>(&saveMatricesTxtFiles)->default_value(saveMatricesTxtFiles),
         "Save projections and intrinsics information in text files.")
        ("rangeStart", po::value<int>(&rangeStart)->default_value(rangeStart),
         "Range image index start.")
        ("rangeSize", po::value<int>(&rangeSize)->default_value(rangeSize),
         "Range size.")
        ("evCorrection", po::value<bool>(&evCorrection)->default_value(evCorrection),
         "Correct exposure value.")
        ("fakeFov", po::value<double>(&fakeFov)->default_value(fakeFov), "Virtual field of view to use if the camera is not pinhole");
    // clang-format on

    CmdLine cmdline("AliceVision prepareDenseScene");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // set output file type
    image::EImageFileType outputFileType = image::EImageFileType_stringToEnum(outImageFileTypeName);

    // Create output dir
    if (!utils::exists(outFolder))
    {
        fs::create_directory(outFolder);
    }

    // Read the input SfM scene
    SfMData sfmData;
    if (!sfmDataIO::load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" << sfmDataFilename << "' cannot be read.");
        return EXIT_FAILURE;
    }

    const double fakeFovRadians = degreeToRadian(fakeFov);
    SfMData outputSfmData;
    if (!convertSfmData(sfmData, outputSfmData, fakeFovRadians, outFolder, outputFileType))
    {
        ALICEVISION_LOG_ERROR("Can't create output sfmData");
        return EXIT_FAILURE;
    }


    //Use transformed sfmData
    int rangeEnd = outputSfmData.getViews().size();

    // set range
    if (rangeStart != -1)
    {
        if (rangeStart < 0 || rangeSize < 0)
        {
            ALICEVISION_LOG_ERROR("Range is incorrect");
            return EXIT_FAILURE;
        }

        if (rangeStart + rangeSize > outputSfmData.getViews().size())
        {
            rangeSize = outputSfmData.getViews().size() - rangeStart;
        }

        rangeEnd = rangeStart + rangeSize;

        if (rangeSize <= 0)
        {
            ALICEVISION_LOG_WARNING("Nothing to compute.");
            return EXIT_SUCCESS;
        }
    }
    else
    {
        rangeStart = 0;
    }


    // export
    if (!prepareDenseScene(sfmData,
                          outputSfmData,
                          imagesFolders,
                          masksFolders,
                          maskExtension,
                          rangeStart,
                          rangeEnd,
                          outFolder,
                          outputFileType,
                          fakeFovRadians,
                          saveMetadata,
                          saveMatricesTxtFiles,
                          evCorrection))
    {
        return EXIT_FAILURE;
    }


    //Only write the sfmData for the first chunk
    if (rangeStart == 0)
    {
        if (!sfmDataIO::save(outputSfmData, sfmDataOutputFilename, sfmDataIO::ESfMData::ALL))
        {
            ALICEVISION_LOG_ERROR("Error writing sfmData");
            return EXIT_FAILURE;
        }
    }

    

    return EXIT_SUCCESS;
}
