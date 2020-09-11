/*
Copyright 2010 Marijn Kentie
 
This file is part of GASW.

GASW is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GASW is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GASW.  If not, see <http://www.gnu.org/licenses/>.
*/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "stdafx.h"
#include "main.h"
#include <wchar.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <string.h>
#include <fstream>
#include <cstddef>
#include <vector>
#include <assert.h>
#include <malloc.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

Options options;


int main(int argc, char* argv[])
{
    
    int numWorkGroup,numWorkItemsperGroup;
    
    options.gapWeight = -10;
    options.extendWeight = -2;
    options.listSize = 20;
    options.dna = 0;
    
    printf("\nSTART OF THE PROGRAM\n");
        

    //Show help info if not enough command line options provided
    if(argc != 4)
    {
        puts("Usage: gpu <blosum> <sequence> <database>");
        puts("Sequence: FASTA format file; Database: DBCONV .gpudb file.");
        return EXIT_SUCCESS;
    }

    options.matrix = argv[1];
    options.sequenceFile = argv[2];
    options.dbFile = argv[3];

    //Show settings that will be used
    printf("Matrix: %s\nSequence: %s\nDatabase: %s\n",options.matrix,options.sequenceFile,options.dbFile);  
    
    // Check Device Capabilities
//--------------------------------------------------------- 

    cl_int status;

    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------
    timestamp_t clinit = get_timestamp();
    
    
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    
    // Use clGetPlatformIDs() to retrieve the number of 
    // platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // Allocate enough space for each platform
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    
    if(status != CL_SUCCESS){
        printf("error in step 1\n");
        exit(-1);
    }
    
//    printf("\nNumber of platform in our environment : %d \n\n",numPlatforms);

//--------------------------------------------------------- 
    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //-----------------------------------------------------

    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    // Use clGetDeviceIDs() to retrieve the number of 
    // devices present
    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL, 
        0, 
        NULL, 
        &numDevices);
    
//    printf("\nNumber of device in our environment : %d \n\n",numDevices);
        
    // Allocate enough space for each device
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    // Fill in devices with clGetDeviceIDs()
    status |= clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL,        
        numDevices, 
        devices, 
        NULL);
    // select the device which will be used
    int device_id = 0;
    
    if((status != CL_SUCCESS) || (device_id >= numDevices)){
        printf("error in step 2\n");
        exit(-1);
    }
    
//---------------------------------------------------------
    //-----------------------------------------------------
    // STEP 3: Create a context
    //-----------------------------------------------------
    cl_context context = NULL;

    // Create a context using clCreateContext() and 
    // associate it with the devices
    context = clCreateContext(
        NULL, 
        1, 
        &devices[device_id], 
        NULL, 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 3\n");
        exit(-1);
    }
    
//---------------------------------------------------------   
    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //-----------------------------------------------------
    
    cl_command_queue cmdQueue;

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute 
    // on
    cmdQueue = clCreateCommandQueue(
        context, 
        devices[device_id], 
        CL_QUEUE_PROFILING_ENABLE,//0, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 4\n");
        exit(-1);
    }
    
    timestamp_t clend = get_timestamp();
    timestamp_t clinittime = clend - clinit;
    //printf("CLInit: %llu\n",clinittime);  

    // Print essential attributes of GPU devices
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;
    size_t maxWorkGroup;
    size_t maxWorkItemDimension[3];
    cl_ulong device_local_size;
    
    for (int j = 0; j < numDevices; j++) {

        // print device name
        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
        printf("%d. Device: %s \n", j+1, value);
        free(value);

        // print hardware device version
        clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
        printf(" %d.%d Hardware version: %s \n", j+1, 1, value);
        free(value);

        // print software driver version
        clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
        printf(" %d.%d Software version: %s \n", j+1, 2, value);
        free(value);

        // print c version supported by compiler for device
        clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
        printf(" %d.%d OpenCL C version: %s \n", j+1, 3, value);
        free(value);

        // print parallel compute units
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(maxComputeUnits), &maxComputeUnits, NULL);
        printf(" %d.%d Parallel compute units: %d \n", j+1, 4, maxComputeUnits);
        
        // print maximum work group size
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(maxWorkGroup), &maxWorkGroup, NULL);
        printf(" %d.%d Maximum Work Group Size: %d \n", j+1, 5, maxWorkGroup);

        // print maximum work item dimensions
        clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                sizeof(size_t)*3, maxWorkItemDimension, NULL);
        printf(" %d.%d Maximum Work Item Dimension: %d %d %d \n", j+1, 6, maxWorkItemDimension[0], maxWorkItemDimension[1], maxWorkItemDimension[2]);
        
        clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE,
                sizeof(device_local_size), &device_local_size, NULL);
        printf(" %d.%d Device Local Memory Size: %lu bytes \n", j+1, 7, device_local_size);
        
        clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(device_local_size), &device_local_size, NULL);
        printf(" %d.%d Device Global Memory Size: %lu bytes \n", j+1, 8, device_local_size);
        
    }
    printf("\n");   

    // Set the work group size and work items size
    numWorkGroup = maxComputeUnits*4;
    numWorkItemsperGroup = 1024/8;

//---------------------------------------------------------
    //-----------------------------------------------------
    // STEP 5: Load query, database and substitution matrix
    //-----------------------------------------------------
    timestamp_t dbinit = get_timestamp();

    //-----------------------------------------------------
    // STEP 5.1: Load query
    //-----------------------------------------------------
    
    const char *fileNamequ = options.sequenceFile;
    bool dna = options.dna;
    const char* ALPHABET;
    if(dna)
        ALPHABET = NUCLEOTIDES;
    else
        ALPHABET = AMINO_ACIDS;

    std::ifstream filequ;
    filequ.open(fileNamequ,std::ios::binary);
    if(!filequ.is_open())
    {   
        printf("Cannot open sequence file: %s\n", fileNamequ);
        return EXIT_FAILURE;
    }
    //Get size
    unsigned int fsize;
    filequ.seekg(0, std::ios::end);
    fsize = filequ.tellg();

    if(fsize < 1) //Empty file
    {
        filequ.close();
        printf("Sequence file empty\n");
        return EXIT_FAILURE;
    }

    //Read file into memory
    buffer = new char[fsize+1];
    filequ.seekg(0);
    filequ.read(buffer,fsize);
    buffer[fsize]= '\0';
    filequ.close();
    if(filequ.bad())
    {
        printf("reading sequence file error\n");
        return EXIT_FAILURE;
    }
    //Process records
    FastaRecord r;
    r.length = 0;
    //char* context;
    char* tokStr=buffer+1;//Skip initial '>'
    while(1)
    {
        r.description=strtok(tokStr,"\n");      
        r.sequence=strtok('\0',">");
        if(!r.description || !r.sequence)
            break;
        records.push_back(r);
        tokStr='\0';
    }   

    //Strip unwanted characters
    for(size_t i=0;i<records.size();i++)
    {

        char* badChar;
        //Strip newlines from description
        while((badChar=strpbrk(records[i].description,"\r\n")))
            *badChar='\0';

        int copyAmt = 0;

        //For each bad character, increase the shift amount so we don't have to perform any superfluous copies
        size_t recLen = strlen(records[i].sequence)+1;
        for(char* c=records[i].sequence;c<records[i].sequence+recLen;c++) //+1 so NULL gets copied
        {
            *c=toupper(*c);
            const char* badResidue;
            if(!dna)
            {
                if(*c!='\0'&&(badResidue=strchr(UNSUPPORTED_LETTERS,*c))!='\0') //Replace unsupported symbols by replacements
                {
                    *c=UNSUPPORTED_LETTERS_REPLACEMENTS[badResidue-UNSUPPORTED_LETTERS];
                }
            }

            const char* residueIndex;

            if((residueIndex=strchr(ALPHABET,*c))=='\0') //Invalid character, skip it
            {
                copyAmt--;
                if(*c!='\n' && *c!='\r' && *c != ' ') //Usually the unsupported characters should only be whitespace and newlines
                {
                    printf("Deleted unknown character %c\n",*c);
                }
            }

            else
            {
                if(*c!='\0')
                {
                    records[i].length++;
                    numSymbols++;
                    *c=residueIndex-ALPHABET; //Replace symbol with its alphabetic index
                }
                if(copyAmt!=0)
                    *(c+copyAmt)=*c;
            }
        }
    }
    
    //////// BELOW TWO VARIABLES ARE SET TO CHOOSE THE TESTED SEQUENCE ////////
    size_t SequenceLength = getSequenceLength(0);
    char* seq = getSequence(0);
    
    printf("%s: input sequence length is %zd.\n",options.sequenceFile,SequenceLength);

//---------------------------------------------------------------
    //-----------------------------------------------------------
    // STEP 5.2: Load substitution matrix
    //-----------------------------------------------------------

    const char *fileNamebm = options.matrix;
    if(!fileNamebm)
    {
        puts("No blosum File\n");
        return EXIT_FAILURE;
    }
    std::ifstream filebm;
    filebm.open(fileNamebm);
    if(!filebm.is_open())
    {
        printf("Cannot open blosum File: %s\n", fileNamebm);
        return EXIT_FAILURE;
    }           
    bool readHeader = false;
    std::string header;
    char row=' ';
    size_t column=0;
    while(1)
    {
        char c;
        filebm >> c;
        if(filebm.eof())
            break;

        //Skip comments
        if(c=='#')
        {           
            filebm.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
            continue;
        }
        
        if(!readHeader) //Read row of amino acid letters
        {           
            if(header.find(c)!=std::string::npos) //Duplicate character: have gone past header into rows
                readHeader=true;
            else                
                header.append(1,c);
        }
        
        if(readHeader) //Interpret rows of values
        {
            if(header.find(c)!=std::string::npos) //Amino
            {
                row=c;
                column=0;
            }
            else //Value
            {               
                if(header.length()<column+1)
                {
                    puts("Error in sub matrix\n");
                    return EXIT_FAILURE;
                }
                char columnName = header.at(column);
                column++;
                filebm.putback(c);
                int val;
                filebm>>val;
                mapMatrix[row][columnName]=val;
            }
        }
            
    }
    
    filebm.close();
//-----------------------------------------------------------------
    //-------------------------------------------------------------
    // STEP 5.3: Create queryprofile 
    //-------------------------------------------------------------
    
    if(dna)
        ALPHABET = NUCLEOTIDES;
    else
        ALPHABET = AMINO_ACIDS;

    size_t seqLength = SequenceLength;//getSequenceLength(0);
    
    queryProfileLength = WHOLE_AMOUNT_OF(SequenceLength,sizeof(queryType))*sizeof(queryType);
    //printf("queryProfileLength : %d \n", queryProfileLength);
    
    size_t qpSize = queryProfileLength*NUM_AMINO_ACIDS;
    queryProfile = new substType[qpSize];
    
    if(!queryProfile)
    {
        puts("queryProfile error\n");
        return EXIT_FAILURE;
    }
    for(size_t j=0;j<NUM_AMINO_ACIDS;j++)
    {
        char d = ALPHABET[j];
        for(size_t i=0;i<queryProfileLength;i++)
        {
            substType s;
            if(i>=seqLength) //If query sequence is too short, pad profile with zeroes
                s=0;
            else
            {
                char q = ALPHABET[seq[i]];
                s =mapMatrix[q][d];
            }
            queryProfile[j*queryProfileLength+i] = s;
        }
    }

//------------------------------------------------------------
    //--------------------------------------------------------
    // STEP 5.4: Load database
    //--------------------------------------------------------
    
    const char *fileNamedb = options.dbFile;
    //printf("Start\n");
    std::ifstream filedb;
    filedb.open(fileNamedb,std::ios::binary);
    if(!filedb.is_open())
    {
        printf("Error reading database file: %s\n", fileNamedb);
        return EXIT_FAILURE;
    }
    //Get metadata
    filedb.read((char*)&metadata,sizeof(metadata)); 

    size_t offsetsSize = metadata.numSequences*sizeof(size_t);
    size_t headerSize = sizeof(metadata)+offsetsSize;
     
    //Get sequence offsets
    sequenceOffsets = new size_t[metadata.numSequences]; 
    //printf("seq\n");
    filedb.read((char*)sequenceOffsets,offsetsSize);

    filedb.seekg(0, std::ios::end);
    blobSize = (size_t)filedb.tellg()-headerSize;
    filedb.seekg(0, std::ios::beg);

    //Read file into memory, skipping the metadata to guarantee alignment
    filedb.seekg(headerSize);
    //printf("1\n");
    blob = (char*)malloc(blobSize);
    filedb.read(blob,blobSize);
    filedb.close();

    blockOffsets = (blockOffsetType*) blob;
    seqNums = (seqNumType*) ((char*) blockOffsets + metadata.numBlocks*sizeof(blockOffsetType));
    seqNums = (seqNumType*) ((char*) seqNums + metadata.alignmentPadding1);
    sequences = (seqType*) ((char*) seqNums + metadata.numBlocks*BLOCK_SIZE*sizeof(seqNumType));
    sequences = (seqType*) ((char*) sequences + metadata.alignmentPadding2);


//----------------------------------------------------------------------
    //------------------------------------------------------------------
    // STEP 5.5: load discriptions
    //------------------------------------------------------------------

    char descsFile[FILENAME_MAX];
    strcpy(descsFile,options.dbFile);
    strcat(descsFile,".descs");
    static const int DESC_SIZE = 100;
    std::ifstream fileds;
    fileds.open(descsFile);
    if(!fileds.is_open())
    {
        printf("Cannot open descrription file: %s\n", descsFile);
        return EXIT_FAILURE;
    }
    descriptions.resize(metadata.numSequences); 
    descriptionBuffer = new char[metadata.numSequences*DESC_SIZE];
    char* ptr = descriptionBuffer;
    for(size_t i=0;i<metadata.numSequences;i++)
    {
        if(fileds.eof())
        {
            printf("Error parsing description file: %s\n", descsFile);
            return EXIT_FAILURE;
        }
        fileds.getline(ptr,DESC_SIZE);
        if(fileds.fail()) //Buffer full
        {
            fileds.clear();
            fileds.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        }
        descriptions[i]=ptr;
        ptr+=DESC_SIZE;
    }
    
 
    timestamp_t dbstop = get_timestamp();
    timestamp_t dbinittime =(dbstop-dbinit);
    //printf("dbInit: %llu\n",dbinittime);
    
    // Debugging     
    printf("%s: %lu symbols in %lu sequence(s) in %lu block(s) in database.\n",options.dbFile,getNumSymbols(),getNumSequences(),getNumBlocks());
   
//--------------------------------------------------------------------
    //----------------------------------------------------------------
    // STEP 6: Copy to GPU
    //----------------------------------------------------------------
    timestamp_t seconds;
    timestamp_t cpstart = get_timestamp();
    //----------------------------------------------------------------
    // STEP 6.1: Copy queryprofile
    //----------------------------------------------------------------
    
    unsigned int columns = WHOLE_AMOUNT_OF(queryProfileLength,sizeof(cl_char4));//queryProfileLength;//
    unsigned int rows = NUM_AMINO_ACIDS;
    
    printf("COLUMNS for query profile: %d\n",columns);
    printf("ROWS for query profile: %d\n",rows);
    
    cl_image_format cif;
    cif.image_channel_order= CL_RGBA;
    cif.image_channel_data_type= CL_SIGNED_INT8;//CL_FLOAT;//
    
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;//CL_MEM_OBJECT_IMAGE1D;
    desc.image_width = columns;//*rows;
    desc.image_height = rows;//1;//
    desc.image_depth = 0;
    desc.image_array_size = 1;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;
    
    cl_mem imgQueryProfile;

    imgQueryProfile = clCreateImage(context,      //context object
                          CL_MEM_READ_ONLY,       //default flags
                          &cif,                   //image format
                          &desc,                  //image description
                          NULL,                   //host pointer
                          &status);               //pointer to error
    
    if(status != CL_SUCCESS){
        printf("error in step 6.1, creating image buffer for query profile\n");
        exit(-1);
    }
    
    size_t origin[3]= {0, 0, 0};//{0, 0, 0};
    size_t region[3]= {0, 0, 0};
    
    region[0]= columns;//*rows;
    region[1]= rows;//1;//rows;//
    region[2]= 1;  
    status |= clEnqueueWriteImage(cmdQueue,                //command queue object
                             imgQueryProfile,                //object to write
                             CL_TRUE,                 //blocking write
                             origin,                  //origin of write
                             region,                  //size of region to write
                             0,                       //row pitch
                             0,                       //slice pitch
                             queryProfile, //  queryProfile_float,//             //data to write
                             0,                       //number of events in the array
                             NULL,                    //array of events
                             NULL);                //output event
    
    if(status != CL_SUCCESS){
        printf("error in step 6.1, writing query profile to image buffer \n");
        exit(-1);
    }
    
//--------------------------------------------------------------------    
    //----------------------------------------------------------------
    // STEP 6.2: Copy database to GPU
    //----------------------------------------------------------------
    /////////// Copy block offsets to GPU ///////////
    
    cl_mem bufferBlockOffsets;
    cl_uint blockOffsetSize = getNumBlocks()*sizeof(blockOffsetType);//+metadata.alignmentPadding1;
    
    bufferBlockOffsets = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        blockOffsetSize, 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 6.2, creating buffer for buffer for Block Offsets\n");
        exit(-1);
    } 
    
    status |= clEnqueueWriteBuffer ( 
        cmdQueue,
        bufferBlockOffsets,
        CL_FALSE,
        0,
        blockOffsetSize,
        blockOffsets,
        0,
        NULL,
        NULL);
        
    if(status != CL_SUCCESS){
        printf("error in step 6.2, Copy block offsets to GPU\n");
        exit(-1);
    } else 
        printf("Copy block offsets to GPU succeed \n");
    
    /////////// Copy sequence Number to GPU ///////////
    
    cl_mem bufferSeqNums;
    cl_uint seqNumSize = getNumBlocks()*BLOCK_SIZE*sizeof(seqNumType);//+metadata.alignmentPadding2;
    
    bufferSeqNums = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        seqNumSize, 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 6.2, creating buffer for Sequence Number\n");
        exit(-1);
    }
    
    status |= clEnqueueWriteBuffer ( 
        cmdQueue,
        bufferSeqNums,
        CL_FALSE,
        0,
        seqNumSize,
        seqNums,
        0,
        NULL,
        NULL);
        
    if(status != CL_SUCCESS){
        printf("error in step 6.2, Copy Sequence Number to GPU\n");
        exit(-1);
    } else 
        printf("Copy bSequence Number to GPU succeed \n");
    
    /////////// Copy Sequences to GPU ///////////
    
    cl_mem bufferSequence;
    uint sequenceSize = blockOffsets[361] + blockOffsets[1];//metadata.numSymbols;//
    
    bufferSequence = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        sequenceSize, 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 6.2, creating buffer for Sequences\n");
        exit(-1);
    }
    
    status |= clEnqueueWriteBuffer ( 
        cmdQueue,
        bufferSequence,
        CL_FALSE,
        0,
        sequenceSize,
        sequences,
        0,
        NULL,
        NULL);
        
    if(status != CL_SUCCESS){
        printf("error in step 6.2, Copy Sequence to GPU\n");
        exit(-1);
    } else 
        printf("Copy Sequence to GPU succeed \n");
      
//--------------------------------------------------------------------  
    //----------------------------------------------------------------
    // STEP 6.3: Create Temporary score matrices and copy to GPU
    //----------------------------------------------------------------
    
    size_t queryLengthDiv2InChunks = WHOLE_AMOUNT_OF(seqLength/2,sizeof(queryType));
   	int matrixSize = queryLengthDiv2InChunks * sizeof(queryType) * sizeof(TempData2); //Size of temporary storage for one thread
    
    printf("Affine Gap matrix size : %d \n",matrixSize);
    printf("Number of Workgroup and Workitem pergroup : %d & %d\n",numWorkGroup,numWorkItemsperGroup);
           
//--------------------------------------------------------------------  
    //----------------------------------------------------------------
    // STEP 6.4: Prepare Output array host and device
    //----------------------------------------------------------------
    
    scoreType* scores=0;
    unsigned int scoreArraySize = sizeof(scoreType)*metadata.numSequences; 
    scores = (scoreType*)malloc(scoreArraySize);

    cl_mem bufferScore;

    bufferScore = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE,                         
        scoreArraySize, 
        NULL, 
        &status);

    if(status != CL_SUCCESS){
        printf("error in step 6.4, creating score buffer is succeed\n");
        exit(-1);
    } else 
        printf("Prepare score array succeed\n");

    timestamp_t cpend = get_timestamp();
    seconds = cpend - cpstart;
    //printf("Copy Time: %llu\n", seconds);

//---------------------------------------------------------------------
    //-----------------------------------------------------------------
    // STEP 7: Create and compile the program
    //-----------------------------------------------------------------
    
    char *kernelFileName, *kernelBuffer;
    kernelFileName = "kernel.cl";
    FILE *kernelFile;
    kernelFile = fopen(kernelFileName, "r");
    if(kernelFile == NULL){
        printf("cannot open .cl file\n");
        printf("current path: %s\n", kernelFileName);
        exit(-1);
    }
    fseek(kernelFile, 0, SEEK_END);
    size_t kernelSize = ftell(kernelFile);
    rewind(kernelFile);

    // read kernel source into buffer
    kernelBuffer = (char*) malloc(kernelSize + 1);
    kernelBuffer[kernelSize] = '\0';
    fread(kernelBuffer, sizeof(char), kernelSize, kernelFile);
    fclose(kernelFile);
    
    timestamp_t argsstart = get_timestamp();

    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**) &kernelBuffer,                                 
        &kernelSize, 
        &status);
    free(kernelBuffer);

    // Build (compile) the program for the devices with
    // clBuildProgram()
    const char compile[] = "-cl-std=CL1.2";
    status |= clBuildProgram(
        program, 
        1, 
        &devices[device_id], 
        compile, 
        NULL, 
        NULL);

    if(status != CL_SUCCESS){
        printf("error in step 7");
        printf(" : %s \n",get_error_string(status));
        exit(-1);
    } else 
        printf("Create and compile the program succeed\n");

//--------------------------------------------------------------------- 
    //-----------------------------------------------------------------
    // STEP 8: Create Kernel
    //-----------------------------------------------------------------
    
    cl_kernel swKernel = NULL;

    // Use clCreateKernel() to create a kernel from the 
    swKernel = clCreateKernel(program, "smithWaterman", &status);
    if(status != CL_SUCCESS){
        printf("error in step 8");
        printf(" : %s \n",get_error_string(status));
        exit(-1);
    } else 
        printf("Create kernel succeed \n");

//---------------------------------------------------------------------
    //-----------------------------------------------------------------
    // STEP 9: Set Kernel Arguments
    //-----------------------------------------------------------------
    
    // Associate the input and output buffers with the 
    // kernel 
    // using clSetKernelArg()
    
    cl_int numGroups = BLOCK_SIZE*getNumBlocks();
    status  = clSetKernelArg(
        swKernel, 
        0, 
        sizeof(cl_int), 
        &numGroups);
      
    status |= clSetKernelArg(
        swKernel, 
        1, 
        sizeof(cl_mem), 
        &bufferScore);
    
    status |= clSetKernelArg(
        swKernel, 
        2, 
        sizeof(cl_mem), 
        &bufferBlockOffsets);
    
    status |= clSetKernelArg(
        swKernel, 
        3, 
        sizeof(cl_mem), 
        &bufferSeqNums);
        
    status |= clSetKernelArg(
        swKernel, 
        4, 
        sizeof(cl_mem), 
        &bufferSequence);

    status |= clSetKernelArg(
        swKernel, 
        5,
        sizeof(cl_mem), 
        &imgQueryProfile); 

    cl_int queryLength = WHOLE_AMOUNT_OF(SequenceLength,sizeof(queryType)); 
    printf("query length : %d\n", queryLength);
    
    status |= clSetKernelArg(
        swKernel, 
        6,
        sizeof(cl_int), 
        &queryLength);

    cl_float gapPenaltyTotal = (cl_float)(options.gapWeight + options.extendWeight); 
    status |= clSetKernelArg(
        swKernel, 
        7,
        sizeof(cl_float), 
        &gapPenaltyTotal);

    cl_float gapExtendPenalty = (cl_float)(options.extendWeight); 
    status |= clSetKernelArg(
        swKernel, 
        8,
        sizeof(cl_float),
        &gapExtendPenalty);
        
    if(status != CL_SUCCESS){
        printf("error in step 9 : ");
        printf("%s \n",get_error_string(status));
        exit(-1);
    } else 
        printf("Kernel Arguments is set successfully\n");

    timestamp_t argsend = get_timestamp();
    timestamp_t argstime = argsend - argsstart;
    printf("OpenCL initialisation Time: %llu\n",clinittime);
    printf("matrix, query and database initialisation Time (host): %llu\n",dbinittime);
    printf("Mem Copy from host to device Time: %llu\n", seconds);
    printf("Time for creating program and setting kernel args: %llu\n", argstime);
    
//-----------------------------------------------------------------------
    //-------------------------------------------------------------------
    // STEP 10: Start the kernel
    //-------------------------------------------------------------------
    //Time the kernel yourself (look at OpenCL profiling)     
    
    // Ensure to have executed all enqueued tasks
    clFinish(cmdQueue); 
    
    // Define an index space (global work size) of work 
    // items for 
    // execution. A workgroup size (local work size) is not 
    // required, 
    // but can be used.
    
    cl_event swDone;
    
    size_t globalWorkSize[1];    
    
    globalWorkSize[0] = numWorkGroup*numWorkItemsperGroup*2;
    
    size_t localWorkSize[1];
    localWorkSize[0] = numWorkItemsperGroup*2;
    
    status |= clEnqueueNDRangeKernel(
        cmdQueue, 
        swKernel, 
        1, 
        NULL, 
        globalWorkSize, 
        localWorkSize, 
        0, 
        NULL, 
        &swDone);

    if(status != CL_SUCCESS){
        clWaitForEvents(1,&swDone);
        printf("error in clEnqueueNDRangeKernel\n");
        exit(-1);
    } else {
        printf("Kernel successfully executed! \n");
    }  
    
    // Ensure kernel execution is finished
    clWaitForEvents(1,&swDone);
    
    cl_ulong time_start, time_end;
    cl_double total_time;
    
    clGetEventProfilingInfo(swDone, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(swDone, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    
    total_time = (cl_double)(time_end - time_start)*(cl_double)(1e-06);
    
    printf("\nKernel computation Time (in ms) = %0.3f ms\n",total_time);
   	
    double numCells = SequenceLength*(double)getNumSymbols();
    printf("Done. Seconds: %f, GCUPS: %f\n\n",total_time, numCells*1000/total_time/(1000*1000*1000));

//------------------------------------------------------------------------
    //--------------------------------------------------------------------
    // STEP 11: Copy results back to host
    //--------------------------------------------------------------------

    clEnqueueReadBuffer(
        cmdQueue, 
        bufferScore, 
        CL_TRUE, 
        0, 
        scoreArraySize, 
        scores, 
        1, 
        &swDone, 
        NULL);

    if(status != CL_SUCCESS){
        printf("error in reading data\n");
        exit(-1);
    } else 
        printf("Score array successfully copied back to CPU\n");

//-------------------------------------------------------------------------
    //---------------------------------------------------------------------
    // Sort scores and print results
    //---------------------------------------------------------------------
    
    printf("\n");
    printf("Protein Alignment OpenCL Implementation\n");
    puts("Sorting results..."); 
    std::vector<Result> sortScores;
    sortScores.resize(getNumSequences());
    for(size_t i=0;i<sortScores.size();i++)
    {
        sortScores[i].index = i;
        sortScores[i].score = scores[i];
    }
    free(scores);
    std::sort(sortScores.begin(),sortScores.end(),&resultComparisonFunc);
    
    //Display results
    puts("Results:");
    for(size_t i=0;i < std::min(20,(int)getNumSequences());i++)  //(int)options.listSize
    {
        printf("%3ld. %-50.50s\t SCORE: %d\n",i,getDescription(sortScores[i].index),sortScores[i].score);
    }

//---------------------------------------------------------------------------
    //-----------------------------------------------------------------------
    // STEP 12: Free OpenCL and C buffers
    //-----------------------------------------------------------------------
    
    // Free OpenCL resources
    clReleaseKernel(swKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferScore);
    clReleaseMemObject(bufferBlockOffsets);
    clReleaseMemObject(bufferSeqNums);
    clReleaseMemObject(bufferSequence);
    clReleaseMemObject(imgQueryProfile);
    clReleaseContext(context);
    
    free(blob);
    
    delete[] queryProfile;
    delete[] sequenceOffsets;
    delete[] descriptionBuffer;
    delete[] buffer;

    return EXIT_SUCCESS;
}

static bool resultComparisonFunc(Result r1, Result r2)
{
    return (r1.score>r2.score);
}


static timestamp_t get_timestamp()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


bool writeToFile(size_t sequenceNum)
{
    if(strcmp(descriptions[sequenceNum],PADDING_SEQ_NAME) == 0)
        return true;
    outFile << '>' << descriptions[sequenceNum] << std::endl;

    size_t offset = sequenceOffsets[sequenceNum];
    
    seqType* ptr = sequences+offset;
            
    while(*ptr!=SUB_SEQUENCE_TERMINATOR&&*ptr!=SEQUENCE_GROUP_TERMINATOR)
    {
        for(size_t i=0;i<SUBBLOCK_SIZE;i++)
        {
            seqType val = ptr[i];
            if(val== ' ') //Subgroup padding
                break;
            outFile << AMINO_ACIDS[val];            
        }
        ptr+=BLOCK_SIZE*SUBBLOCK_SIZE;
    }
    outFile << '\n';
    return true;
}


size_t getSequenceLength(size_t sequenceNum)
{
    if(sequenceNum >= records.size())
        return 0;

    return records[sequenceNum].length;
}

char* getSequence(size_t sequenceNum)
{
    if(sequenceNum >= records.size())
        return NULL;
    
    return records[sequenceNum].sequence;

}

size_t getNumSequences()
{
    return metadata.numSequences;
}

size_t getDBSizeInBytes()
{
    return blobSize;
}

size_t getNumSymbols()
{
    return metadata.numSymbols;
}

size_t getNumBlocks()
{
    return metadata.numBlocks;
}

const char* getDescription(unsigned int index)
{
    if(index>=descriptions.size())
        return NULL;
    return descriptions[index];
}

const char *get_error_string(cl_int error)
{
    switch (error) {
        // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}


