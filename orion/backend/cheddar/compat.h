#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct ArrayResultInt {
  int *Data;
  unsigned long Length;
};

struct ArrayResultFloat {
  float *Data;
  unsigned long Length;
};

struct ArrayResultDouble {
  double *Data;
  unsigned long Length;
};

struct ArrayResultUInt64 {
  unsigned long long *Data;
  unsigned long Length;
};

struct ArrayResultByte {
  unsigned char *Data;
  unsigned long Length;
};

void NewScheme(int logN, const int *logQ, int lenQ, const int *logP, int lenP,
               int logScale, int h, const char *ringType, const char *keysPath,
               const char *ioMode);
void DeleteScheme();
void FreeCArray(void *ptr);

void NewKeyGenerator();
void GenerateSecretKey();
void GeneratePublicKey();
void GenerateRelinearizationKey();
void GenerateEvaluationKeys();
ArrayResultByte SerializeSecretKey();
void LoadSecretKey(const unsigned char *data, unsigned long lenData);

void NewEncoder();
int Encode(const float *values, int lenValues, int level,
           unsigned long scale);
ArrayResultFloat Decode(int plaintextID);
ArrayResultDouble DecodeComplex(int plaintextID);

void NewEncryptor();
void NewDecryptor();
int Encrypt(int plaintextID);
int Decrypt(int ciphertextID);

void DeletePlaintext(int plaintextID);
void DeleteCiphertext(int ciphertextID);
unsigned long GetPlaintextScale(int plaintextID);
double GetPlaintextScaleLog2(int plaintextID);
unsigned long GetCiphertextScale(int ciphertextID);
double GetCiphertextScaleLog2(int ciphertextID);
void SetPlaintextScale(int plaintextID, unsigned long scale);
void SetCiphertextScale(int ciphertextID, unsigned long scale);
int GetPlaintextLevel(int plaintextID);
int GetCiphertextLevel(int ciphertextID);
int GetPlaintextSlots(int plaintextID);
int GetCiphertextSlots(int ciphertextID);
int GetCiphertextDegree(int ciphertextID);
ArrayResultUInt64 GetModuliChain();
ArrayResultUInt64 GetAuxModuliChain();
ArrayResultUInt64 GetDeviceMemoryInfo();
void SynchronizeDevice();
void TrimDeviceMemoryPool(unsigned long long targetBytes);
double ConsumeDeviceMemoryTrimSeconds();
ArrayResultDouble ConsumeSharedCacheEvalProfileSeconds();
ArrayResultInt GetLivePlaintexts();
ArrayResultInt GetLiveCiphertexts();

void NewEvaluator();
void AddRotationKey(int rotation);
int Negate(int ciphertextID);
int Conjugate(int ciphertextID);
int ConjugateNew(int ciphertextID);
int Rotate(int ciphertextID, int amount);
int RotateNew(int ciphertextID, int amount);
int Rescale(int ciphertextID);
int RescaleNew(int ciphertextID);
int AddScalar(int ciphertextID, float scalar);
int AddScalarNew(int ciphertextID, float scalar);
int SubScalar(int ciphertextID, float scalar);
int SubScalarNew(int ciphertextID, float scalar);
int MulScalarInt(int ciphertextID, int scalar);
int MulScalarIntNew(int ciphertextID, int scalar);
int MulScalarFloat(int ciphertextID, float scalar);
int MulScalarFloatNew(int ciphertextID, float scalar);
int MulImaginaryUnit(int ciphertextID, int sign);
int MulImaginaryUnitNew(int ciphertextID, int sign);
int AddPlaintext(int ciphertextID, int plaintextID);
int AddPlaintextNew(int ciphertextID, int plaintextID);
int SubPlaintext(int ciphertextID, int plaintextID);
int SubPlaintextNew(int ciphertextID, int plaintextID);
int MulPlaintext(int ciphertextID, int plaintextID);
int MulPlaintextNew(int ciphertextID, int plaintextID);
int AddCiphertext(int lhsID, int rhsID);
int AddCiphertextNew(int lhsID, int rhsID);
int SubCiphertext(int lhsID, int rhsID);
int SubCiphertextNew(int lhsID, int rhsID);
int MulRelinCiphertext(int lhsID, int rhsID);
int MulRelinCiphertextNew(int lhsID, int rhsID);

void NewPolynomialEvaluator();
int GenerateMonomial(const float *coeffs, int lenCoeffs);
int GenerateChebyshev(const float *coeffs, int lenCoeffs);
int EvaluatePolynomial(int ciphertextID, int polyID, unsigned long outScale);
ArrayResultDouble GenerateMinimaxSignCoeffs(const int *degrees, int lenDegrees,
                                            int prec, int logalpha,
                                            int logerr, int debug);
int GetPolyDepth(int polyID);

void NewLinearTransformEvaluator();
int GenerateLinearTransform(const int *diagIdxs, int diagIdxsLen,
                            const float *diagData, int diagDataLen, int level,
                            float bsgsRatio, const char *ioMode);
int EvaluateLinearTransform(int transformID, int ciphertextID);
void DeleteLinearTransform(int transformID);
ArrayResultInt GetLinearTransformRotationKeys(int transformID);
ArrayResultInt GetLinearTransformRotationKeyRequests(int transformID);
ArrayResultUInt64 EstimateLinearTransformDeviceBytes(int transformID);
int LinearTransformUsesStreaming(int transformID);
void GenerateLinearTransformRotationKey(int key);
void GenerateLinearTransformRotationKeyAtLevel(int key, int level);
ArrayResultInt GenerateLinearTransformsUnified(
    int numTransforms, const int *const *diagIdxsArray, const int *diagIdxsLens,
    const float *const *diagDataArray, const int *diagDataLens,
    const int *levels);
ArrayResultInt GenerateLinearTransformsUnifiedComplex(
    int numTransforms, const int *const *diagIdxsArray, const int *diagIdxsLens,
    const double *const *diagDataArray, const int *diagDataLens,
    const int *levels);
ArrayResultInt EvaluateLinearTransformsWithSharedCache(const int *transformIDs,
                                                       int numTransforms,
                                                       int ciphertextID);
void PrepareLinearTransformsSharedCachePlan(const int *transformIDs,
                                            int numTransforms);
ArrayResultByte GenerateAndSerializeRotationKey(int key);
ArrayResultByte GenerateAndSerializeRotationKeyAtLevel(int key, int level);
void LoadRotationKey(const unsigned char *data, unsigned long lenData,
                     unsigned long key);
void LoadLinearTransformRotationKey(const unsigned char *data,
                                    unsigned long lenData,
                                    unsigned long key,
                                    int transformID);
void RemoveLinearTransformRotationKeys(int transformID);
ArrayResultByte SerializeDiagonal(int transformID, int diagIdx);
ArrayResultByte SerializeLinearTransformPlaintexts(int transformID);
void LoadPlaintextDiagonal(const unsigned char *data, unsigned long lenData,
                          int transformID, unsigned long diagIdx);
void LoadPlaintextDiagonalsBatch(const unsigned char *data,
                                 unsigned long lenData,
                                 const unsigned long long *offsets,
                                 int numOffsets,
                                 const unsigned long long *lengths,
                                 int numLengths, const int *diagIdxs,
                                 int numDiagIdxs, int transformID);
void LoadLinearTransformPlaintexts(const unsigned char *data,
                                   unsigned long lenData,
                                   int transformID);
void RemovePlaintextDiagonals(int transformID);
void ReleaseLinearTransformMatrix(int transformID);
void RemoveRotationKeys();

void NewBootstrapper(const int *logPs, int lenLogPs, int numSlots);
int Bootstrap(int ciphertextID, int numSlots);
void DeleteBootstrappers();

}
