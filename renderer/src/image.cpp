/*
 * image.cpp
 *
 *  Created on: Nov 30, 2015
 *      Author: igkiou
 */

#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#ifdef USE_OPENEXR
#include <ImathBox.h>
#include <ImfRgba.h>
#include <ImfRgbaFile.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfHeader.h>
#include <ImathBox.h>
#include <ImfInputFile.h>
#include <ImfHeader.h>
#include <ImfArray.h>
#include <ImfChannelList.h>
#include <ImfAttribute.h>
#include <ImfStandardAttributes.h>
#include <ImfPixelType.h>
#include <ImfFrameBuffer.h>
#endif

#include "image.h"

namespace image {

#ifdef USE_OPENEXR
void SmallImage::writeOpenEXR(const std::string& fileName) const {

	Imf::Header head((int) m_xRes, (int) m_yRes);
	head.channels().insert("Y", Imf::Channel(kEXRFloatUsed));
	Imf::FrameBuffer frameBuffer;
	frameBuffer.insert("Y", Imf::Slice(kEXRFloatUsed, (char *) m_pixels,
					sizeof(*m_pixels) * 1, sizeof(*m_pixels) * m_xRes));

	Imf::OutputFile file(fileName.c_str(), head);
	file.setFrameBuffer(frameBuffer);
	file.writePixels(m_yRes);
}
#endif

namespace {

Image2<float>::EByteOrder getHostByteOrder() {
	union {
		std::uint8_t  charValue[2];
		std::uint16_t shortValue;
	};
	charValue[0] = 1;
	charValue[1] = 0;

	return (shortValue == 1)?(Image2<float>::EByteOrder::ELittleEndian)
							:(Image2<float>::EByteOrder::EBigEndian);
}

} /* namespace */

template <>
void Image2<float>::writePFM(const std::string& fileName) const {//,
//						const EByteOrder fileEndianness) const {
	Assert((fileEndianness == EBigEndian) || (fileEndianness == ELittleEndian));

	std::ofstream file(fileName.c_str(),
					std::ofstream::out | std::ofstream::binary);
	AssertEx(file, "Problem writing output file.");

	file << "Pf";
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");
	file << m_xRes;
	AssertEx(file, "Problem writing output file.");
	file << ' ';
	AssertEx(file, "Problem writing output file.");
	file << m_yRes;
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");
	file << ((getHostByteOrder() == Image2<float>::EByteOrder::ELittleEndian)
							?(static_cast<float>(-1.0))
							:(static_cast<float>(1.0)));
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");

    Float* img = new Float[m_xRes*m_yRes];
	for (int i = 0; i < m_yRes; i++) {
		memcpy(&img[(m_yRes - i - 1)*(m_xRes)],
			&m_pixels[(i*(m_xRes))],
			(m_xRes) * sizeof(Float));
	}

	// Always save pfm's as double
	double* double_pixels = new double[m_xRes*m_yRes];
	for (size_t i=0; i < m_xRes*m_yRes; ++i) {
		double_pixels[i] = static_cast<double>(img[i]);
	}

	file.write(reinterpret_cast<char*>(double_pixels),
			m_xRes * m_yRes * sizeof(double));

//  Old code
//	file.write(reinterpret_cast<char*>(m_pixels),
//			m_xRes * m_yRes * sizeof(float));

	AssertEx(file, "Problem writing output file.");
	delete[] img;
	delete[] double_pixels;
}

template <>
void Image3<Float>::writePFM(const std::string& fileNamePrefix) const {//,
//						const EByteOrder fileEndianness) const {

//	Assert((fileEndianness == EBigEndian) || (fileEndianness == ELittleEndian));

	std::stringstream ofname;

	for(int z=0; z<m_zRes; z++){
		ofname.str("");
		ofname << fileNamePrefix << "_" << z << ".pfm";
		std::ofstream file(ofname.str().c_str(),
						std::ofstream::out | std::ofstream::binary);
		AssertEx(file, "Problem writing output file.");

		file << "Pf";
		AssertEx(file, "Problem writing output file.");
		file << '\n';
		AssertEx(file, "Problem writing output file.");
		file << m_xRes;
		AssertEx(file, "Problem writing output file.");
		file << ' ';
		AssertEx(file, "Problem writing output file.");
		file << m_yRes;
		AssertEx(file, "Problem writing output file.");
		file << '\n';
		AssertEx(file, "Problem writing output file.");
		file << ((getHostByteOrder() == Image2<Float>::EByteOrder::ELittleEndian)
								?(static_cast<Float>(-1.0))
								:(static_cast<Float>(1.0)));
		AssertEx(file, "Problem writing output file.");
		file << '\n';
		AssertEx(file, "Problem writing output file.");

		file.write(reinterpret_cast<char*>( m_pixels + z * m_xRes * m_yRes),
				m_xRes * m_yRes * sizeof(Float));
		AssertEx(file, "Problem writing output file.");
	}
}


template <>
void Image3<Float>::writePFM3D(const std::string& fileName) const {//,
//						const EByteOrder fileEndianness) const {

//	Assert((fileEndianness == EBigEndian) || (fileEndianness == ELittleEndian));


	std::ofstream file(fileName.c_str(),
					std::ofstream::out | std::ofstream::binary);
	AssertEx(file, "Problem writing output file.");

	file << "Pf";
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");
	file << m_xRes;
	AssertEx(file, "Problem writing output file.");
	file << ' ';
	AssertEx(file, "Problem writing output file.");
	file << m_yRes;
	file << ' ';
	AssertEx(file, "Problem writing output file.");
	file << m_zRes;
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");

	file << ((getHostByteOrder() == Image2<Float>::EByteOrder::ELittleEndian)
							?(static_cast<Float>(-1.0))
							:(static_cast<Float>(1.0)));
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");

	double *double_pixels = new double[m_xRes * m_yRes * m_zRes];
	for (size_t i = 0; i < m_xRes * m_yRes * m_zRes; ++i) {
		double_pixels[i] = static_cast<double>(m_pixels[i]);
	}

	for(int z = 0; z < m_zRes; z++)
		file.write(reinterpret_cast<char*>(double_pixels + z * m_xRes * m_yRes),
				m_xRes * m_yRes * sizeof(double));
	AssertEx(file, "Problem writing output file.");

	delete[] double_pixels;
}

}	/* namespace image */
