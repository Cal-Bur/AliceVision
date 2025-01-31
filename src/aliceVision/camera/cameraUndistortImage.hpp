// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/system/Logger.hpp>
#include <aliceVision/image/Image.hpp>
#include <aliceVision/image/Sampler.hpp>
#include <aliceVision/camera/cameraCommon.hpp>
#include <aliceVision/camera/IntrinsicBase.hpp>
#include <aliceVision/camera/IntrinsicScaleOffsetDisto.hpp>
#include <aliceVision/camera/Pinhole.hpp>
#include <aliceVision/camera/Undistortion.hpp>
#include <aliceVision/image/io.hpp>

#include <memory>

namespace aliceVision {
namespace camera {

/**
 * @Brief undistort an image when both intrinsic may be of different camera types
 * @param imageIn the distorted image
 * @param intrinsicSource the camera of the distorted image
 * @param intrinsicOutput the camera of the undistorted image
 * @param image_ud output image
 * @param fillColor what is the default color when the pixel does not exists in the source image
 * @param roi an optional roi for the output
 */
template<typename T>
void undistortImage(const image::Image<T>& imageIn,
                    const camera::IntrinsicBase & intrinsicSource,
                    const camera::IntrinsicBase & intrinsicOutput,
                    image::Image<T>& image_ud,
                    T fillcolor,
                    const oiio::ROI& roi = oiio::ROI());

/**
 * @Brief undistort an image
 * @param imageIn the distorted image
 * @param intrinsicPtr the camera of the image
 * @param image_ud output image
 * @param fillColor what is the default color when the pixel does not exists in the source image
 * @param correctPrincipalPoint do we want to correct the image to compensate the principal point shift
 * @param roi an optional roi for the output
 */
template<typename T>
void undistortImage(const image::Image<T>& imageIn,
                    const camera::IntrinsicBase* intrinsicPtr,
                    image::Image<T>& image_ud,
                    T fillcolor,
                    bool correctPrincipalPoint = false,
                    const oiio::ROI& roi = oiio::ROI());


template<typename T>
void undistortImage(const image::Image<T>& imageIn,
                    const camera::IntrinsicBase & intrinsicSource,
                    const camera::IntrinsicBase & intrinsicOutput,
                    image::Image<T>& image_ud,
                    T fillcolor,
                    const oiio::ROI& roi)
{
    // There is distortion
    const Vec2 center(imageIn.width() * 0.5, imageIn.height() * 0.5);

    int widthRoi = intrinsicOutput.w();
    int heightRoi = intrinsicOutput.h();
    int xOffset = 0;
    int yOffset = 0;
    if (roi.defined())
    {
        widthRoi = roi.width();
        heightRoi = roi.height();
        xOffset = roi.xbegin;
        yOffset = roi.ybegin;
    }

    image_ud.resize(widthRoi, heightRoi, true, fillcolor);
    const image::Sampler2d<image::SamplerLinear> sampler;

#pragma omp parallel for
    for (int y = 0; y < heightRoi; ++y)
    {
        for (int x = 0; x < widthRoi; ++x)
        {
            const Vec2 undisto_pix(x + xOffset, y + yOffset);

            // compute coordinates with distortion
            const Vec3 intermediate = intrinsicOutput.backProjectUnit(undisto_pix);
            const Vec2 disto_pix = intrinsicSource.project(intermediate.homogeneous(), true);

            // pick pixel if it is in the image domain
            if (imageIn.contains(disto_pix(1), disto_pix(0)))
            {
                image_ud(y, x) = sampler(imageIn, disto_pix(1), disto_pix(0));
            }
        }
    }
}

template<typename T>
void undistortImage(const image::Image<T>& imageIn,
                    const camera::IntrinsicBase* intrinsicPtr,
                    image::Image<T>& image_ud,
                    T fillcolor,
                    bool correctPrincipalPoint,
                    const oiio::ROI& roi)
{
    if (!intrinsicPtr->hasDistortion())  // no distortion, perform a direct copy
    {
        image_ud = imageIn;
        return;
    }

    // There is distortion
    const Vec2 center(imageIn.width() * 0.5, imageIn.height() * 0.5);
    Vec2 ppCorrection(0.0, 0.0);

    if (correctPrincipalPoint)
    {
        if (camera::isPinhole(intrinsicPtr->getType()))
        {
            const camera::Pinhole* pinholePtr = dynamic_cast<const camera::Pinhole*>(intrinsicPtr);
            ppCorrection = pinholePtr->getPrincipalPoint() - center;
        }
    }

    int widthRoi = imageIn.width();
    int heightRoi = imageIn.height();
    int xOffset = 0;
    int yOffset = 0;
    if (roi.defined())
    {
        widthRoi = roi.width();
        heightRoi = roi.height();
        xOffset = roi.xbegin;
        yOffset = roi.ybegin;
    }

    image_ud.resize(widthRoi, heightRoi, true, fillcolor);
    const image::Sampler2d<image::SamplerLinear> sampler;

#pragma omp parallel for
    for (int y = 0; y < heightRoi; ++y)
    {
        for (int x = 0; x < widthRoi; ++x)
        {
            const Vec2 undisto_pix(x + xOffset, y + yOffset);
            // compute coordinates with distortion
            const Vec2 disto_pix = intrinsicPtr->getDistortedPixel(undisto_pix + ppCorrection);

            // pick pixel if it is in the image domain
            if (imageIn.contains(disto_pix(1), disto_pix(0)))
            {
                image_ud(y, x) = sampler(imageIn, disto_pix(1), disto_pix(0));
            }
        }
    }
}

}  // namespace camera
}  // namespace aliceVision
