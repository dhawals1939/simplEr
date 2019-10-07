#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <vector>

typedef double Float;

#ifdef NDEBUG
#define Assert(cond) ((void) 0)
#define AssertEx(cond, explanation) ((void) 0)
#else
/* Assertions */
// Assert that a condition is true
#define Assert(cond) do { \
		if (!(cond)) fprintf(stderr, "Assertion \"%s\" failed in %s:%i\n", \
		#cond, __FILE__, __LINE__); \
	} while (0)

// Assertion with a customizable error explanation
#define AssertEx(cond, explanation) do { \
                if (!(cond)) fprintf(stderr, "Assertion \"%s\" failed in %s:%i (" explanation ")\n", \
                                        #cond, __FILE__, __LINE__); \
            } while (0)
#endif

enum EByteOrder {
	EBigEndian = 0,
	ELittleEndian = 1,
	EByteOrderLength = 2,
	EByteOrderInvalid = -1
};

EByteOrder getHostByteOrder() {
	union {
		std::uint8_t  charValue[2];
		std::uint16_t shortValue;
	};
	charValue[0] = 1;
	charValue[1] = 0;

	return (shortValue == 1)?(EByteOrder::ELittleEndian)
							:(EByteOrder::EBigEndian);
}


void readPFM(Float* &pixels, int& xRes, int& yRes, const std::string& filename){
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
			xRes = atoi(c);
			fscanf(pFile, "%s", c);
			yRes = atoi(c);
			int length_ = xRes * yRes;
			fscanf(pFile, "%s", c);
			Float endianess = atof(c);

			fseek(pFile, 0, SEEK_END);
			long lSize = ftell(pFile);
			long pos = lSize - xRes*yRes * sizeof(Float);
			fseek(pFile, pos, SEEK_SET);

			Float* img = new Float[length_];
			//cout << "sizeof(T) = " << sizeof(T);
			fread(img, sizeof(Float), length_, pFile);
			fclose(pFile);

			/* The raster is a sequence of pixels, packed one after another,
			 * with no delimiters of any kind. They are grouped by row,
			 * with the pixels in each row ordered left to right and
			 * the rows ordered bottom to top.
			 */
			pixels = (Float  *)malloc(length_ * sizeof(Float));// top-to-bottom.
			//PFM SPEC image stored bottom -> top reversing image
			for (int i = 0; i < yRes; i++) {
				memcpy(&pixels[(yRes - i - 1)*(xRes)],
					&img[(i*(xRes))],
					(xRes) * sizeof(Float));
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

void writePFM(Float* &pixels, const int& xRes, const int& yRes, const std::string& fileName) {//,

	std::ofstream file(fileName.c_str(), std::ofstream::out | std::ofstream::binary);
	AssertEx(file, "Problem writing output file.");

	file << "Pf";
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");
	file << xRes;
	AssertEx(file, "Problem writing output file.");
	file << ' ';
	AssertEx(file, "Problem writing output file.");
	file << yRes;
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");
	file << ((getHostByteOrder() == EByteOrder::ELittleEndian)
							?(static_cast<Float>(-1.0))
							:(static_cast<Float>(1.0)));
	AssertEx(file, "Problem writing output file.");
	file << '\n';
	AssertEx(file, "Problem writing output file.");
			
    Float* img = new Float[xRes*yRes];
	for (int i = 0; i < yRes; i++) {
		memcpy(&img[(yRes - i - 1)*(xRes)],
			&pixels[(i*(xRes))],
			(xRes) * sizeof(Float));
	}

	file.write(reinterpret_cast<char*>(img),
			xRes * yRes * sizeof(Float));
	AssertEx(file, "Problem writing output file.");
}

/* tokenize similar to mitsuba, to make input parsing more intuititive */
std::vector<std::string> tokenize(const std::string &string, const std::string &delim) {
	std::string::size_type lastPos = string.find_first_not_of(delim, 0);
	std::string::size_type pos = string.find_first_of(delim, lastPos);
	std::vector<std::string> tokens;

	while (std::string::npos != pos || std::string::npos != lastPos) {
		tokens.push_back(string.substr(lastPos, pos - lastPos));
		lastPos = string.find_first_not_of(delim, pos);
		pos = string.find_first_of(delim, lastPos);
	}

	return tokens;
}


int main(int argc, char **argv){

    std::string prefix = "../renderings/complexTiming";
    int renderings = 2;
    int tBins      = 4096;

	for(int i = 1; i < argc; i++){
		std::vector<std::string> param = tokenize(argv[i], "=");
		if(param.size() != 2){
			std::cerr << "Input argument " << argv[i] << "should be in the format arg=value" << std::endl;
			return -1;
		}
		if(param[0].compare("prefix")==0)
			prefix = param[1];
		else if(param[0].compare("renderings")==0)
			renderings = stoi(param[1]);
		else if(param[0].compare("tBins")==0)
			tBins = stoi(param[1]);
		else{
			std::cerr << "Unknown variable in the input argument:" << param[0] << std::endl;
			std::cerr << "Should be one of "
					  << "projectorTexture "
					  << "projectorTexture "
					  << "projectorTexture "
					  << std::endl;
			return -1;
        }
    }


    for(int i = 1; i < argc; i++){
        std::vector<std::string> param = tokenize(argv[i], "=");
        if(param[0].compare("prefix") == 0)
            prefix = param[1];
        else if(param[0].compare("renderings") == 0)
            renderings = stoi(param[1]);
        else if(param[0].compare("tBins") == 0)
            tBins = stoi(param[1]);
        else{
			std::cerr << "Unknown variable in the input argument:" << param[0] << std::endl;
			std::cerr << "Should be one of "
					  << "prefix "
					  << "renderings "
					  << "tBins "
					  << std::endl;
			return -1;
        }
    }

   
    Float *mean_pixels[tBins];
    Float *tmp_pixels[tBins];
    int xRes;
    int yRes;
    for(int j = 0; j < tBins; j++){
        readPFM(mean_pixels[j], xRes, yRes, prefix + "_0_" + std::to_string(j) + ".pfm");
    }


    for(int j = 0; j < tBins; j++){
        for(int i = 1; i < renderings; i++){
            readPFM(tmp_pixels[j], xRes, yRes, prefix + "_" + std::to_string(i) + "_" + std::to_string(j) + ".pfm");
            for (int temp = 0; temp < xRes*yRes; temp++)
                *(mean_pixels[j]+temp) = (*(mean_pixels[j]+temp)*(i+1) + *(tmp_pixels[j]+temp))/(i+2) ;
        }
    }

    for(int j = 0; j < tBins; j++){
        writePFM(mean_pixels[j], xRes, yRes, prefix + "_mean_" + std::to_string(j) + ".pfm");
    }

    return 0;
}
