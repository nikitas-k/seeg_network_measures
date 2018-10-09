import itk
import sys

def canny3d(inputfile, outputfile, variance=0.0, lthres=0.0, uthres=1.0)
    
inputImage = inputfile
outputImage = outputfile
variance = float(variance)
lowerThreshold = float(lthres)
upperThreshold = float(uthres)

InputPixelType = itk.F
OutputPixelType = itk.UC
Dimension = 3

InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]

reader = itk.ImageFileReader[InputImageType].New()
reader.SetFileName(inputImage)

cannyFilter = itk.CannyEdgeDetectionImageFilter[
    InputImageType,
    InputImageType].New()
cannyFilter.SetInput(reader.GetOutput())
cannyFilter.SetVariance(variance)
cannyFilter.SetLowerThreshold(lowerThreshold)
cannyFilter.SetUpperThreshold(upperThreshold)

rescaler = itk.RescaleIntensityImageFilter[
    InputImageType,
    OutputImageType].New()

rescaler.SetInput(cannyFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[OutputImageType].New()
writer.SetFileName(outputImage)
writer.SetInput(rescaler.GetOutput())

writer.Update()
