//Add your kernel function here

#define WHOLE_AMOUNT_OF(x,y) ((x+y-1)/y)

typedef char substType;

typedef uint blockOffsetType;
typedef uint seqSizeType;
typedef uint seqNumType;
typedef char seqType;
typedef char8 seqType8;

typedef ushort scoreType;
typedef ushort4 scoreType4;
typedef ushort8 scoreType8;

typedef char4 queryType ;

__constant size_t BLOCK_SIZE = 16; /**< How many sequences make up a block. Should be device half-warp size. Should be power of 2. */
__constant size_t LOG2_BLOCK_SIZE = 4; /**< To be able to use a shift instead of division in the device code */
__constant size_t ALIGNMENT = 64;
__constant size_t SUBBLOCK_SIZE = 8; /**< The number of symbols of each sequence to interleave at a time; to be able to read multiple symbols in one access */
__constant char SEQUENCE_GROUP_TERMINATOR = ' ';
__constant char SUB_SEQUENCE_TERMINATOR = '#';

constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;

/**
Align 4 query profile entries with a database residue.
*/
void alignResidues(float *maxScore, float4 *left, float4 *ixLeft, float *top, float *topLeft, float *IyTop, float4 *substScore, float c_gapPenaltyTotal, float c_gapExtendPenalty)
{	
  //q0
	(*ixLeft).x = fmax(0.0f,fmax((*left).x+c_gapPenaltyTotal,(*ixLeft).x+c_gapExtendPenalty)); //max(0,...) here so IxColumn[] can be unsigned
	(*IyTop) = fmax((*top)+c_gapPenaltyTotal,(*IyTop)+c_gapExtendPenalty);
	float align = (*topLeft)+(*substScore).x;
	(*topLeft) = (*left).x;
	(*left).x = fmax(align,fmax((*ixLeft).x,(*IyTop)));
		
	//q1
	(*ixLeft).y = fmax(0.0f,fmax((*left).y+c_gapPenaltyTotal,(*ixLeft).y+c_gapExtendPenalty)); //max(0,...) here so IxColumn[] can be unsigned
	(*IyTop) = fmax((*left).x+c_gapPenaltyTotal,(*IyTop)+c_gapExtendPenalty);
	align = (*topLeft)+(*substScore).y;
	(*topLeft) = (*left).y;
	(*left).y = fmax(align,fmax((*ixLeft).y,(*IyTop)));

	//q2
	(*ixLeft).z = fmax(0.0f,fmax((*left).z+c_gapPenaltyTotal,(*ixLeft).z+c_gapExtendPenalty)); //max(0,...) here so IxColumn[] can be unsigned
	(*IyTop) = fmax((*left).y+c_gapPenaltyTotal,(*IyTop)+c_gapExtendPenalty);
	align = (*topLeft)+(*substScore).z;
	(*topLeft)=(*left).z;
	(*left).z = fmax(align,fmax((*ixLeft).z,(*IyTop)));

	//q3
	(*ixLeft).w = fmax(0.0f,fmax((*left).w+c_gapPenaltyTotal,(*ixLeft).w+c_gapExtendPenalty)); //max(0,...) here so IxColumn[] can be unsigned
	(*IyTop) = fmax((*left).z+c_gapPenaltyTotal,(*IyTop)+c_gapExtendPenalty);
	align = (*topLeft)+(*substScore).w;	
	(*left).w = fmax(align,fmax((*ixLeft).w,(*IyTop)));

	(*topLeft)=(*top); //The next column is to the right of this one, so current top left becomes new top
	(*top) = (*left).w; //Set top value for next query chunk
	(*maxScore) = fmax((*left).x,fmax((*left).y,fmax((*left).z,fmax((*left).w,(*maxScore))))); //Update max score
}

/**
Align a database sequence subblock with the entire query sequence.
The loading/aligning with the query sequence in the 'inner' function as query sequence (constant) memory is much faster than the global memory in which the db resides.
*/ 
void alignWithQuery(seqType8 s, float column, float4* tempColumn, float *maxScore, __read_only image2d_t queryProfile, int c_queryLength, float c_gapPenaltyTotal, float c_gapExtendPenalty)
{
	//Set the top related values to 0 as we're at the top of the matrix
	float8 top = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	float topLeft = 0.0f;
	float8 IyTop = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};		

	float4 substScores; //Query profile scores
	float4 left;
	float4 ixLeft;
  
  float temp;
  float temp2;
  
	for(int j = 0; j < c_queryLength; j++)
	{
		//Load first half of temporary column
		float4 t = tempColumn[0]; 
    left.x = column*t.x;
		ixLeft.x = column*t.y;
		left.y = column*t.z;
		ixLeft.y = column*t.w;

		//Load second half of temporary column
		t = tempColumn[1];
    left.z = column*t.x;
		ixLeft.z = column*t.y; 
		left.w = column*t.z; 
		ixLeft.w = column*t.w; 

		float topLeftNext = left.w; //Save the top left cell value for the next loop interation
		
    //d0
    substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s0)));
    temp = top.s0;
    temp2 = IyTop.s0;
		alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s0 = temp;
    IyTop.s0 = temp2;
   
		//d1
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s1)));
    temp = top.s1;
    temp2 = IyTop.s1;		
    alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s1 = temp;
    IyTop.s1 = temp2;
    
		//d2
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s2)));
    temp = top.s2;
    temp2 = IyTop.s2;				
    alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s2 = temp;
    IyTop.s2 = temp2;

		//d3
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s3)));
    temp = top.s3;
    temp2 = IyTop.s3;				
		alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s3 = temp;
    IyTop.s3 = temp2;

		//d4
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s4)));
    temp = top.s4;
    temp2 = IyTop.s4;		
		alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s4 = temp;
    IyTop.s4 = temp2;

		//d5
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s5)));
    temp = top.s5;
    temp2 = IyTop.s5;	
		alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s5 = temp;
    IyTop.s5 = temp2;

		//d6
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s6)));
    temp = top.s6;
    temp2 = IyTop.s6;	
		alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s6 = temp;
    IyTop.s6 = temp2;

		//d7
    temp = top.s7;
    temp2 = IyTop.s7;	
		substScores = convert_float4_rtp(read_imagei(queryProfile,sampler,(int2)(j,s.s7))); 
		alignResidues(maxScore, &left, &ixLeft, &temp, &topLeft, &temp2, &substScores,c_gapPenaltyTotal,c_gapExtendPenalty);
    top.s7 = temp;
    IyTop.s7 = temp2;


		topLeft = topLeftNext;
		
		//Save the two temporary column values
		t.x = left.x; 
    t.y = ixLeft.x; 
    t.z = left.y; 
    t.w = ixLeft.y; 
		tempColumn[0]=t;
		tempColumn++;
   
    t.x = left.z; 
		t.y = ixLeft.z; 
    t.z = left.w; 
    t.w = ixLeft.w;
		tempColumn[0]=t;	
    tempColumn++;
	}
}

///**
//Align the database sequence subblocks with the query sequence until a terminating subblock is encountered.
//*/ /**/
void align(__constant seqType* sequence, float4* tempColumn, seqNumType* seqNum, __global scoreType* scores,__read_only image2d_t queryProfile, int c_queryLength, float c_gapPenaltyTotal, float c_gapExtendPenalty) 
{
	float maxScore=0.0f;
  float column = 0.0f;
  
	seqType8 s = vload8(0,sequence);
 
  seqSizeType temp_seqNum = *seqNum; 
  
  while(s.s0!=' ')//Until terminating subblock
	{		
		if(s.s0=='#') //Subblock signifying concatenated sequences
		{ 
      scores[temp_seqNum] = maxScore; 
      temp_seqNum++;
      column=maxScore=0.0f;
		}

		alignWithQuery(s,column,tempColumn,&maxScore,queryProfile,c_queryLength,c_gapPenaltyTotal,c_gapExtendPenalty);
   
		column=1.0f;
		sequence += BLOCK_SIZE*SUBBLOCK_SIZE;	
    s = vload8(0,sequence);
	}
	
	scores[temp_seqNum] = convert_ushort_sat_rte(maxScore); //Set score for sequence
  (*seqNum) = temp_seqNum;
  
	return;
}

/**
Main kernel function
*/
__kernel void smithWaterman(int numGroups, __global scoreType* scores, __constant blockOffsetType* blockOffsets, __constant seqNumType* seqNums, __constant seqType* sequences, __read_only image2d_t queryProfile, int c_queryLength, float c_gapPenaltyTotal, float c_gapExtendPenalty)
{
	int groupNum = get_global_id(0);	
  __private float4 tempColumn[1024];
  
  while(groupNum<numGroups)
	{
		seqNumType seqNum=seqNums[groupNum];	//Get sequence number of first sequence in group
    
		// GPUdb::LOG2_BLOCK_SIZE : 4 To be able to use a shift instead of division in the device code 
    int seqBlock = groupNum >> LOG2_BLOCK_SIZE; // divided by 2^4 -> defines the location group of groupNum 
		int groupNumInBlock = groupNum & (BLOCK_SIZE-1); //equals groupNum % GPUdb::BLOCK_SIZE -> define the location groupNum in seqBlock
		int groupOffset = blockOffsets[seqBlock]+mul24(groupNumInBlock,SUBBLOCK_SIZE); // SUBBLOCK_SIZE = 8 The number of symbols of each sequence to interleave at a time; to be able to read multiple symbols in one access 

     align(&sequences[groupOffset],tempColumn,&seqNum,scores,queryProfile,c_queryLength,c_gapPenaltyTotal,c_gapExtendPenalty); //Perform alignment 

    groupNum+= get_num_groups(0)*get_local_size(0);
	}
}