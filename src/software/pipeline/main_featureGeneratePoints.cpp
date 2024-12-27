// This file is part of the AliceVision project.
// Copyright (c) 2024 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/system/main.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/cmdline/cmdline.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <aliceVision/matchingImageCollection/GeometricFilterType.hpp>
#include <aliceVision/matchingImageCollection/pairBuilder.hpp>
#include <aliceVision/matchingImageCollection/ImagePairListIO.hpp>

#include <aliceVision/robustEstimation/ACRansac.hpp>
#include <aliceVision/robustEstimation/estimators.hpp>
#include <aliceVision/multiview/relativePose/Fundamental7PSolver.hpp>
#include <aliceVision/multiview/relativePose/FundamentalError.hpp>
#include <aliceVision/multiview/RelativePoseKernel.hpp>
#include <aliceVision/multiview/Unnormalizer.hpp>

#include <aliceVision/sfm/pipeline/regionsIO.hpp>


#include <aliceVision/matching/matchesFiltering.hpp>
#include <aliceVision/matching/io.hpp>

#include <aliceVision/image/io.hpp>

#include <aliceVision/matching/IndMatch.hpp>
#include <aliceVision/sfm/pipeline/relativeConstraints.hpp>

#include <boost/program_options.hpp>

#include <aliceVision/track/TracksBuilder.hpp>
#include <aliceVision/track/tracksUtils.hpp>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

namespace po = boost::program_options;

using namespace aliceVision;

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string sfmDataFilename;
    std::string estimationFolder;
    std::string matchesFolderOutput;
    std::string imagesFolder;
    std::vector<std::string> predefinedPairList;
    int rangeStart = -1;
    int rangeSize = 0;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
             "SfMData file.")
        ("output,o", po::value<std::string>(&matchesFolderOutput)->required(),
            "Path to a folder in which computed matches will be stored.")
        ("imagesFolder", 
            po::value<std::string>(&imagesFolder)->required(),
            "Path to folder(s) containing the extracted features.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("rangeStart", 
            po::value<int>(&rangeStart)->default_value(rangeStart),
            "Range image index start.")
        ("rangeSize", 
            po::value<int>(&rangeSize)->default_value(rangeSize),
            "Range size.");

    // clang-format on

    CmdLine cmdline("This program filters input matches according to a geometric model:\n"
                    "AliceVision featureGeometricFiltering");

    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    sfmData::SfMData sfmData;
    if (!sfmDataIO::load(sfmData, sfmDataFilename, 
            sfmDataIO::ESfMData(
                sfmDataIO::VIEWS | sfmDataIO::INTRINSICS | sfmDataIO::EXTRINSICS
            )
        ))
    {
        ALICEVISION_LOG_ERROR("The input SfMData file '" << sfmDataFilename << "' cannot be read.");
        return EXIT_FAILURE;
    }

    //Create pairs
    std::set<Pair> pairs;
    auto keys = sfmData.getViewsKeys();
    for (const auto & key : keys)
    {
        for (const auto & other : keys)
        {
            if (key >= other)
            {
                continue;
            }

            pairs.insert(std::make_pair(key, other));
        }
    }


    matching::PairwiseMatches pwmatches;
    int count = 0;
    for (const auto &pair: pairs)
    {
        const IndexT referenceId = pair.first;
        const IndexT nextId = pair.second;

        std::string pathDirect = imagesFolder + "/" + std::to_string(referenceId) + "_" + std::to_string(nextId)  + "_warp.exr";
        std::string pathIndirect = imagesFolder + "/" + std::to_string(nextId) + "_" + std::to_string(referenceId)  + "_warp.exr";

        if (!(std::filesystem::exists(pathDirect) && std::filesystem::exists(pathIndirect)))
        {
            continue;
        }

        image::Image<image::RGBfColor> warpDirect, warpIndirect;
        image::readImage(pathDirect, warpDirect, image::EImageColorSpace::NO_CONVERSION);
        image::readImage(pathIndirect, warpIndirect, image::EImageColorSpace::NO_CONVERSION);

        matching::IndMatches matches;

        for (int i = 0; i < warpDirect.height(); i++)
        {
            for (int j = 0; j < warpDirect.width(); j++)
            {
                auto & pix = warpDirect(i, j);
                if (pix.b() < 0.1)
                {
                    pix.r() = 0.0f;
                    pix.g() = 0.0f;
                    pix.b() = 0.0f;
                    continue;
                }

                double x = pix.r() * 864.0;
                double y = pix.g() * 864.0;
                int oj = static_cast<int>(std::floor(x));
                int oi = static_cast<int>(std::floor(y));

                double minDist = 2;
                int besti = 0;
                int bestj = 0;

                for (int di = 0; di < 2; di++)
                {
                    for (int dj = 0; dj < 2; dj++)
                    {
                        int ci = oi + di;
                        int cj = oj + dj;

                        if (ci < 0 || cj < 0 || ci >= 864.0 || cj >= 864.0)
                        {
                            continue;
                        }

                        const auto & opix = warpIndirect(ci, cj);
                        if (opix.b() < 0.1)
                        {
                            continue;
                        }

                        double refx = opix.r() * 864.0;
                        double refy = opix.g() * 864.0;

                        double diffx = refx - double(j);
                        double diffy = refy - double(i);

                        double dist = sqrt(diffx*diffx+diffy*diffy);
                        
                        if (dist < minDist)
                        {
                            minDist = dist;
                            besti = ci;
                            bestj = cj;
                        }
                    }
                }

                if (minDist > 1.0)
                {
                    pix.r() = 0.0;
                    pix.g() = 0.0;
                    pix.b() = 0.0;
                }
                else 
                {
                    int id1 = i * 864 + j;
                    int id2 = besti * 864 + bestj;
                    matching::IndMatch match(id1, id2);
                    matches.push_back(match);
                }
            }
        }
        
        std::cout << matches.size() << std::endl;
        pwmatches[pair][feature::EImageDescriberType::SIFT] = matches;
        
        count += matches.size();
    }
    std::cout << pwmatches.size() << std::endl;
    std::cout << count << std::endl;

    ALICEVISION_LOG_INFO("tracks build");
    track::TracksBuilder builder;
    builder.build(pwmatches);
    track::TracksMap allTracks;
    builder.exportToSTL(allTracks, nullptr);

    track::TracksPerView mapTracksPerView;
    track::computeTracksPerView(allTracks, mapTracksPerView);
    ALICEVISION_LOG_INFO("tracks build done");

    int length = 2; 
    for (int length = 0; length < 129; length++)
    {
        int count = 0;
        for (const auto & [id, track] : allTracks)
        {
            if (track.featPerView.size() ==length)
            {
                count++;
            }
        }


        std::cout << length << " : "  << count  << std::endl;
    }

    /*for (const auto & [idView, tracks] : mapTracksPerView)
    {
        std::vector<double> weights;

        for (auto & trackId : tracks)
        {
            weights.push_back(std::exp(allTracks[trackId].featPerView.size()));
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(weights.begin(), weights.end());
    }*/

    return EXIT_SUCCESS;
}
