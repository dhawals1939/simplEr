/*
 * image.h
 *
 *  Created on: Nov 30, 2015
 *      Author: igkiou
 */
#pragma once

#include <cstdint>
#include <malloc.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>

#if USE_OPENEXR
#include <OpenEXR/ImfPixelType.h>
const Imf::PixelType kEXRFloatUsed = Imf::FLOAT;
#endif

#include <constants.h>
#include <util.h>
#include <tvector.h>

namespace image {

/*
 * TODO: Replace the image data and access operations with ipp ones, for faster
 * parallelism.
 * TODO: Add version for reference return, or unary operators to single pixels.
 */
template <typename T>
class Image2 {
public:
	enum EFileFormat {
		EOpenEXR = 0,
		EPFM = 1,
		EFileFormatLength = 2,
		EFileFormatInvalid = -1
	};

	enum EByteOrder {
		EBigEndian = 0,
		ELittleEndian = 1,
		EByteOrderLength = 2,
		EByteOrderInvalid = -1
	};

	Image2() :
			m_xRes(0),
			m_yRes(0),
			m_pixels(NULL) {	}

	Image2(const int xRes, const int yRes) :
			m_xRes(xRes),
			m_yRes(yRes),
			m_pixels(NULL) {
			Assert(xRes > 0 && yRes > 0);
		m_pixels = (T *) memalign(L1_CACHE_LINE_SIZE,
									m_xRes * m_yRes * sizeof(T));
	}

	void setSize(const int xRes, const int yRes) {
		Assert(m_xRes == 0 && m_yRes == 0 && m_pixels == NULL);
		Assert(xRes > 0 && yRes > 0);
		m_xRes = xRes;
		m_yRes = yRes;
		m_pixels = (T *) malloc(m_xRes * m_yRes * sizeof(T));
	}

	inline void zero() {
		memset((void *) m_pixels, 0, m_xRes * m_yRes * sizeof(T));
	}

	//similar to MATLAB implementation
	inline void ind2sub(const int &ndx, int &x, int &y) const {
		Assert(ndx >= 0 && ndx < m_xRes*m_yRes);
		y = ndx/m_xRes;
		x = ndx - y*m_xRes;
	}

	//similar to MATLAB implementation
	inline void sub2ind(const int &x, const int &y, int &ndx) const {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes);
		ndx = y * m_xRes + x;
	}

	inline T getPixel(const int x) const {
		Assert(x >= 0 && x < m_xRes*m_yRes);
		return m_pixels[x];
	}

	inline T getPixel(const int x, const int y) const {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes);
		return m_pixels[y * m_xRes + x];
	}

	inline void setPixel(const int x, const int y, const T val) {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes);
//		Assert(val >= 0);
		m_pixels[y * m_xRes + x] = val;
	}

	inline void addEnergy(const int x, const int y, const T val) {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes);
//		Assert(val >= 0);
		m_pixels[y * m_xRes + x] += val;
	}

	inline const T* getImage() const {
		return m_pixels;
	}

	inline int getXRes() const {
		return m_xRes;
	}

	inline int getYRes() const {
		return m_yRes;
	}

#if NDEBUG
	inline void copyImage(T *buffer, const int) const {
#else
	inline void copyImage(T *buffer, const int size) const {
#endif
		Assert(size == m_xRes * m_yRes);
		memcpy((void *) buffer, (void *) m_pixels,
			m_xRes * m_yRes * sizeof(T));
	}


	inline void readFile(const std::string& fileName,
							const EFileFormat fileFormat = EPFM) {
//							const EByteOrder fileEndianness = ELittleEndian)

		switch (fileFormat) {
			case EOpenEXR: {
				readOpenEXR(fileName);
				break;
			}
			case EPFM: {
				readPFM(fileName);
				break;
			}
			case EFileFormatInvalid:
			default: {
				Assert((fileFormat == EPFM) || (fileFormat == EOpenEXR));
				break;
			}
		}
	}


	inline void writeToFile(const std::string& fileName,
							const EFileFormat fileFormat = EPFM) const {
//							const EByteOrder fileEndianness = ELittleEndian)

		switch (fileFormat) {
			case EOpenEXR: {
				writeOpenEXR(fileName);
				break;
			}
			case EPFM: {
				writePFM(fileName);//, fileEndianness);
				break;
			}
			case EFileFormatInvalid:
			default: {
				Assert((fileFormat == EPFM) || (fileFormat == EOpenEXR));
				break;
			}
		}
	}

	~Image2() {
		free(m_pixels);
	}

private:
	void readPFM(const std::string& filename){
		FILE * pFile;
		pFile = fopen(filename.c_str(), "rb");
		char c[100];
		if (pFile != NULL) {
			fscanf(pFile, "%s", c);
			// strcmp() returns 0 if they are equal.
			if (!strcmp(c, "Pf")) {
				fscanf(pFile, "%s", c);
				// atoi: ASCII to integer.
				// itoa: integer to ASCII.
				this->m_xRes = atoi(c);
				fscanf(pFile, "%s", c);
				this->m_yRes = atoi(c);
				int length_ = this->m_xRes * this->m_yRes;
				fscanf(pFile, "%s", c);
				Float endianess = atof(c);

				fseek(pFile, 0, SEEK_END);
				long lSize = ftell(pFile);
				long pos = lSize - this->m_xRes*this->m_yRes * sizeof(double);
				fseek(pFile, pos, SEEK_SET);

				double* img = new double[length_];
				//cout << "sizeof(T) = " << sizeof(T);
				fread(img, sizeof(double), length_, pFile);
				fclose(pFile);

				/* The raster is a sequence of pixels, packed one after another,
				 * with no delimiters of any kind. They are grouped by row,
				 * with the pixels in each row ordered left to right and
				 * the rows ordered bottom to top.
				 */
				double *double_pixels = (double*)malloc(length_ * sizeof(double));// top-to-bottom.
				m_pixels = (T *)malloc(length_ * sizeof(T));
				//PFM SPEC image stored bottom -> top reversing image
				for (int i = 0; i < this->m_yRes; i++) {
					memcpy(&double_pixels[(this->m_yRes - i - 1)*(this->m_xRes)],
						&img[(i*(this->m_xRes))],
						(this->m_xRes) * sizeof(double));
				}

				for (size_t i=0; i < length_; ++i) {
					m_pixels[i] = static_cast<T>(double_pixels[i]);
				}
				delete[] img;
				free(double_pixels);
			}
			else {
				std::cout << "Invalid magic number!"
					<< " No Pf (meaning grayscale pfm) is missing!!\n";
				fclose(pFile);
				exit(0);
			}

		}
		else {
			std::cout << "Cannot open file " << filename
				<< ", or it does not exist!\n";
			fclose(pFile);
			exit(0);
		}
	}
	void writePFM(const std::string& fileName) const; //, const EByteOrder fileEndianness) const;
#if USE_OPENEXR
	void writeOpenEXR(const std::string& fileName) const;
#else
	inline void readOpenEXR(const std::string&) const {
		std::cerr << "Reading openEXR format is not implemented." << std::endl;
	}
	inline void writeOpenEXR(const std::string&) const {
		std::cerr << "Writing to OpenEXR format disabled during compilation." << std::endl;
	}
#endif

	int m_xRes;
	int m_yRes;
	T *m_pixels;
};

template <typename T>
class Image3 {
public:
	enum EFileFormat {
		EOpenEXR = 0,
		EPFM = 1,
		EFileFormatLength = 2,
		EFileFormatInvalid = -1
	};

	enum EByteOrder {
		EBigEndian = 0,
		ELittleEndian = 1,
		EByteOrderLength = 2,
		EByteOrderInvalid = -1
	};

	Image3() :
			m_xRes(0),
			m_yRes(0),
			m_zRes(0),
			m_pixels(NULL) {	}

	Image3(const int xRes, const int yRes, const int zRes) :
			m_xRes(xRes),
			m_yRes(yRes),
			m_zRes(zRes),
			m_pixels(NULL) {
		Assert(xRes > 0 && yRes > 0 && zRes > 0);
		m_pixels = (T *) memalign(L1_CACHE_LINE_SIZE,
									m_xRes * m_yRes * m_zRes * sizeof(T));
	}

	void setSize(const int xRes, const int yRes, const int zRes) {
		Assert(m_xRes == 0 && m_yRes == 0 && m_zRes == 0 && m_pixels == NULL);
		Assert(xRes > 0 && yRes > 0 && zRes > 0);
		m_xRes = xRes;
		m_yRes = yRes;
		m_zRes = zRes;
		m_pixels = (T *) malloc(m_xRes * m_yRes * m_zRes * sizeof(T));
	}

	inline void zero() {
		memset((void *) m_pixels, 0, m_xRes * m_yRes * m_zRes * sizeof(T));
	}

	inline T getPixel(const int x, const int y, const int z) const {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes && z >= 0 && z < m_zRes);
		return m_pixels[z * m_xRes * m_yRes + y * m_yRes + x];
	}

	inline void setPixel(const int x, const int y, const int z, const T val) {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes && z >= 0 && z < m_zRes);
//		Assert(val >= 0);
		m_pixels[z * m_xRes * m_yRes + y * m_yRes + x] = val;
	}

	inline void addEnergy(const int x, const int y, const int z, const T val) {
		Assert(x >= 0 && x < m_xRes && y >= 0 && y < m_yRes && z >= 0 && z < m_zRes);
//		Assert(val >= 0);
		m_pixels[z * m_xRes * m_yRes + y * m_xRes + x] += val;
	}

	inline int getXRes() const {
		return m_xRes;
	}

	inline int getYRes() const {
		return m_yRes;
	}

	inline int getZRes() const {
		return m_zRes;
	}

#if NDEBUG
	inline void copyImage(T *buffer, const int) const {
#else
	inline void copyImage(T *buffer, const int size) const {
#endif
		Assert(size == m_xRes * m_yRes * m_zRes);
		memcpy((void *) buffer, (void *) m_pixels,
			m_xRes * m_yRes * m_zRes * sizeof(T));
	}

	inline void writeToFile(const std::string& fileName,
							const EFileFormat fileFormat = EPFM) const {
//							const EByteOrder fileEndianness = ELittleEndian)

		switch (fileFormat) {
			case EOpenEXR: {
				writeOpenEXR(fileName);
				break;
			}
			case EPFM: {
				writePFM(fileName);//, fileEndianness);
				break;
			}
			case EFileFormatInvalid:
			default: {
				Assert((fileFormat == EPFM) || (fileFormat == EOpenEXR));
				break;
			}
		}
	}

	~Image3() {
		free(m_pixels);
	}

	void writePFM3D(const std::string& fileName) const;
	void readPFM3D(const std::string& filename){
		FILE * pFile;
		pFile = fopen(filename.c_str(), "rb");
		char c[100];
		if (pFile != NULL) {
			fscanf(pFile, "%s", c);
			// strcmp() returns 0 if they are equal.
			if (!strcmp(c, "Pf")) {
				fscanf(pFile, "%s", c);
				// atoi: ASCII to integer.
				// itoa: integer to ASCII.
				this->m_xRes = atoi(c);
				fscanf(pFile, "%s", c);
				this->m_yRes = atoi(c);
				fscanf(pFile, "%s", c);
				this->m_zRes = atoi(c);
				int length_ = this->m_xRes * this->m_yRes * this->m_zRes;
				fscanf(pFile, "%s", c);
				Float endianess = atof(c);

				fseek(pFile, 0, SEEK_END);
				long lSize = ftell(pFile);
				long pos = lSize - this->m_xRes*this->m_yRes*this->m_zRes * sizeof(double);
				fseek(pFile, pos, SEEK_SET);

				double* img = new double[length_];
				//cout << "sizeof(T) = " << sizeof(T);
				fread(img, sizeof(double), length_, pFile);
				fclose(pFile);

				/* The raster is a sequence of pixels, packed one after another,
				 * with no delimiters of any kind. They are grouped by row,
				 * with the pixels in each row ordered left to right and
				 * the rows ordered bottom to top.
				 */
				m_pixels = (Float *)malloc(length_ * sizeof(Float));// top-to-bottom.
				double *double_pixels = (double *)malloc(length_ * sizeof(double));
				//Adi: Did not test exhaustively
//                for (int z = 0; z < this->m_zRes; z++) {
				memcpy(double_pixels,
					   img,
					   length_ * sizeof(double));
//              }
//
				for (size_t i=0; i < length_; ++i) {
					m_pixels[i] = static_cast<float>(double_pixels[i]);
				}

				delete[] img;

			}
			else {
				std::cout << "Invalid magic number!"
					<< " No Pf (meaning grayscale pfm) is missing!!\n";
				fclose(pFile);
				exit(0);
			}

		}
		else {
			std::cout << "Cannot open file " << filename
				<< ", or it does not exist!\n";
			fclose(pFile);
			exit(0);
		}
	}
private:
	void writePFM(const std::string& fileNamePrefix) const;
#if USE_OPENEXR
	void writeOpenEXR(const std::string& fileName) const;
#else
	inline void readOpenEXR(const std::string&) const {
		std::cerr << "Reading openEXR format is not implemented." << std::endl;
	}
	inline void writeOpenEXR(const std::string&) const {
		std::cerr << "Writing to OpenEXR format disabled during compilation." << std::endl;
	}
#endif

	int m_xRes;
	int m_yRes;
	int m_zRes;
	T *m_pixels;
};

template <typename T>
class Image2Set {
public:
	explicit Image2Set(const int numImages)
				: m_xRes(0),
				  m_yRes(0),
				  m_numImages(numImages) {
		m_images = new Image2<T>[m_numImages];
	}

	Image2Set(const int xRes, const int yRes, const int numImages)
				: m_xRes(xRes),
				  m_yRes(yRes),
				  m_numImages(numImages) {
		m_images = new Image2<T>[m_numImages];
		for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
			m_images[iterImage].setSize(m_xRes, m_yRes);
		}
	}

	inline void setSize(const int xRes, const int yRes) {
		Assert(m_xRes == 0 && m_yRes == 0);
		m_xRes = xRes;
		m_yRes = yRes;
		for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
			m_images[iterImage].setSize(m_xRes, m_yRes);
		}
	}

	inline void zero() {
		for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
			m_images[iterImage].zero();
		}
	}

	inline Image2<T> &operator[](int i) {
		return m_images[i];
	}

	inline const Image2<T> &operator[](int i) const {
		return m_images[i];
	}

	inline void mergeImages(Image2<T> &mergedImage) const {
		Assert(mergedImage.getXRes() == m_xRes && mergedImage.getYRes() == m_yRes);
		for (int i = 0; i < m_yRes; ++i) {
			for (int j = 0; j < m_xRes; ++j) {
				T val = 0.0;
				for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
					val += m_images[iterImage].getPixel(j, i);
				}
				mergedImage.setPixel(j, i, val);
			}
		}
	}

	~Image2Set() {
		delete[] m_images;
	}

private:
	int m_xRes;
	int m_yRes;
	int m_numImages;
	Image2<T> *m_images;
};


template <typename T>
class Image3Set {
public:
	explicit Image3Set(const int numImages) :
			m_xRes(0),
			m_yRes(0),
			m_zRes(0),
			m_numImages(numImages) {
		m_images = new Image3<T>[m_numImages];
	}

	Image3Set(const int xRes, const int yRes, const int zRes, const int numImages) :
			m_xRes(xRes),
			m_yRes(yRes),
			m_zRes(zRes),
			m_numImages(numImages) {
		m_images = new Image3<T>[m_numImages];
		for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
			m_images[iterImage].setSize(m_xRes, m_yRes, m_zRes);
		}
	}

	inline void setSize(const int xRes, const int yRes, const int zRes) {
		Assert(m_xRes == 0 && m_yRes == 0);
		m_xRes = xRes;
		m_yRes = yRes;
		m_zRes = zRes;
		for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
			m_images[iterImage].setSize(m_xRes, m_yRes, m_zRes);
		}
	}

	inline void zero() {
		for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
			m_images[iterImage].zero();
		}
	}

	inline Image3<T> &operator[](int i) {
		return m_images[i];
	}

	inline const Image3<T> &operator[](int i) const {
		return m_images[i];
	}

	void mergeImages(Image3<T> &mergedImage) const {
		Assert(mergedImage.getXRes() == m_xRes && mergedImage.getYRes() == m_yRes && mergedImage.getZRes() == m_zRes);
		for (int h = 0; h < m_zRes; ++h) {
			for (int i = 0; i < m_yRes; ++i) {
				for (int j = 0; j < m_xRes; ++j) {
					T val = 0.0;
					for (int iterImage = 0; iterImage < m_numImages; ++iterImage) {
						val += m_images[iterImage].getPixel(j, i, h);
					}
					mergedImage.setPixel(j, i, h, val);
				}
			}
		}
	}

	~Image3Set() {
		delete[] m_images;
	}

private:
	int m_xRes;
	int m_yRes;
	int m_zRes;
	int m_numImages;
	Image3<T> *m_images;
};
typedef Image2<Float> Texture;
typedef Image3<Float> SmallImage;
typedef Image3Set<Float> SmallImageSet;

} /* namespace image */
