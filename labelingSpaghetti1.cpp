// Copyright(c) 2016 - Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
// 
// * Neither the name of YACCLAB nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "labelingSpaghetti1.h"

using namespace cv;
using namespace std;

//Find the root of the tree of node i
//template<typename LabelT>
inline static
uint findRoot(const uint *P, uint i){
	uint root = i;
	while (P[root] < root){
		root = P[root];
	}
	return root;
}

//Make all nodes in the path of node i point to root
//template<typename LabelT>
inline static
void setRoot(uint *P, uint i, uint root){
	while (P[i] < i){
		uint j = P[i];
		P[i] = root;
		i = j;
	}
	P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
//template<typename LabelT>
inline static
uint find(uint *P, uint i){
	uint root = findRoot(P, i);
	setRoot(P, i, root);
	return root;
}

//unite the two trees containing nodes i and j and return the new root
//template<typename LabelT>
inline static
uint set_union(uint *P, uint i, uint j){
	uint root = findRoot(P, i);
	if (i != j){
		uint rootj = findRoot(P, j);
		if (root > rootj){
			root = rootj;
		}
		setRoot(P, j, root);
	}
	setRoot(P, i, root);
	return root;
}

//Flatten the Union Find tree and relabel the components
//template<typename LabelT>
inline static
uint flattenL(uint *P, uint length){
	uint k = 1;
	for (uint i = 1; i < length; ++i){
		if (P[i] < i){
			P[i] = P[P[i]];
		}
		else{
			P[i] = k; k = k + 1;
		}
	}
	return k;
}


inline static
void firstScanSpaghetti1(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);

#define condition_b img_row_prev_prev[c-1]>0
#define condition_c img_row_prev_prev[c]>0
//#define condition_d c+1<w && img_row_prev_prev[c+1]>0
#define condition_d img_row_prev_prev[c+1]>0
#define condition_e img_row_prev_prev[c+2]>0

#define condition_g img_row_prev[c-2]>0
#define condition_h img_row_prev[c-1]>0
#define condition_i img_row_prev[c]>0
//#define condition_j c+1<w && img_row_prev[c+1]>0
#define condition_j img_row_prev[c+1]>0
#define condition_k img_row_prev[c+2]>0

#define condition_m img_row[c-2]>0
#define condition_n img_row[c-1]>0
#define condition_o img_row[c]>0
//#define condition_p c+1<w && img_row[c+1]>0
#define condition_p img_row[c+1]>0

//#define condition_r r+1<h && img_row_fol[c-1]>0
#define condition_r img_row_fol[c-1]>0
//#define condition_s r+1<h && img_row_fol[c]>0
#define condition_s img_row_fol[c]>0
//#define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0
#define condition_t img_row_fol[c+1]>0

#define if_finish_condition	if ((c += 2) >= w - 2)

#define action_1	imgLabels_row[c] = 0; //Action_1: No action (the block has no foreground pixels) 
#define action_2	imgLabels_row[c] = lunique; /*Action_2: New label (the block has foreground pixels and is not connected to anything else) */ \
					P[lunique] = lunique;		\
					lunique = lunique + 1;			
#define action_3	imgLabels_row[c] = imgLabels_row_prev_prev[c - 2]; //Action_3: Assign label of block P
#define action_4	imgLabels_row[c] = imgLabels_row_prev_prev[c]; //Action_4: Assign label of block Q 
#define action_5	imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];	//Action_5: Assign label of block R
#define action_6	imgLabels_row[c] = imgLabels_row[c - 2];	//Action_6: Assign label of block S
#define action_7	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);	//Action_7: Merge labels of block P and Q
#define action_8	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);	//Action_8: Merge labels of block P and R
#define action_9	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);	// ACTION_9 Merge labels of block P and S
#define action_10	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);	// ACTION_10 Merge labels of block Q and R
#define action_11	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);	//Action_11: Merge labels of block Q and S
#define action_12	imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);	//Action_12: Merge labels of block R and S
//Action_13:	// Merge labels of block P, Q and R
//			imgLabels(r,c) = es.resolve(imgLabels(r-2,c-2),imgLabels(r-2,c),imgLabels(r-2,c+2));
//			continue;
#define action_14	imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);	//Action_14: Merge labels of block P, Q and S
#define action_15	imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);	//Action_15: Merge labels of block P, R and S
#define action_16	imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);	//Action_16: labels of block Q, R and S

	for (int r = 0; r < 2; r += 2) {
		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);

		int c = 0;
		if (condition_o) {
			if (condition_p) {
				action_2;
				goto fr_tree_2;
			}
			else {
				action_2;
				goto fr_tree_3;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_0: if_finish_condition goto fr_break_0;
		if (condition_o) {
			if (condition_p) {
				action_2;
				goto fr_tree_2;
			}
			else {
				action_2;
				goto fr_tree_3;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_1: if_finish_condition goto fr_break_1;
		if (condition_o) {
			if (condition_p) {
				action_6;
				goto fr_tree_2;
			}
			else {
				action_6;
				goto fr_tree_3;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
					goto fr_tree_2;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_2: if_finish_condition goto fr_break_2;
		if (condition_o) {
			if (condition_p) {
				action_6;
				goto fr_tree_2;
			}
			else {
				action_6;
				goto fr_tree_3;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
					goto fr_tree_2;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_tree_3: if_finish_condition goto fr_break_3;
		if (condition_o) {
			if (condition_r) {
				if (condition_p) {
					action_6;
					goto fr_tree_2;
				}
				else {
					action_6;
					goto fr_tree_3;
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					action_2;
					goto fr_tree_3;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
						goto fr_tree_2;
					}
					else {
						action_2;
						goto fr_tree_2;
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto fr_tree_3;
					}
					else {
						action_2;
						goto fr_tree_3;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
					goto fr_tree_2;
				}
				else {
					if (condition_t) {
						action_2;
						goto fr_tree_1;
					}
					else {
						action_1;
						goto fr_tree_0;
					}
				}
			}
		}
	fr_break_0:
		if (condition_o) {
			if (condition_p) {
				action_2;
			}
			else {
				action_2;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_2;
				}
				else {
					action_2;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_1:
		if (condition_o) {
			if (condition_p) {
				action_6;
			}
			else {
				action_6;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_2:
		if (condition_o) {
			if (condition_p) {
				action_6;
			}
			else {
				action_6;
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	fr_break_3:
		if (condition_o) {
			if (condition_r) {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					action_2;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					action_2;
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	}
		
	for (int r = 2; r < h; r += 2) {
		// Get rows pointer
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
		const uchar* const img_row_prev_prev = (uchar *)(((char *)img_row_prev) - img.step.p[0]);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
	
		int c = 0;
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					action_4;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_5;
							goto tree_4;
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						action_2;
						goto tree_61;
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_0: if_finish_condition goto break_0;
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
							goto tree_73;
						}
						else {
							action_7;
							goto tree_73;
						}
					}
					else {
						action_4;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_5;
										goto tree_4;
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_8;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_3;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_2;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
						goto tree_63;
					}
					else {
						if (condition_h) {
							action_3;
							goto tree_61;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_2;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_1: if_finish_condition goto break_1;
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
							goto tree_73;
						}
						else {
							action_14;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_i) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_16;
								goto tree_4;
							}
						}
						else {
							if (condition_h) {
								if (condition_d) {
									if (condition_c) {
										action_12;
										goto tree_4;
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_15;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_47;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
						goto tree_82;
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_2: if_finish_condition goto break_2;
		if (condition_o) {
			if (condition_j) {
				action_11;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_12;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_3: if_finish_condition goto break_3;
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_b) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_4: if_finish_condition goto break_4;
		if (condition_o) {
			if (condition_j) {
				action_6;
				goto tree_73;
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							action_6;
							goto tree_4;
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_6;
								goto tree_4;
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_d) {
								action_5;
								goto tree_4;
							}
							else {
								action_10;
								goto tree_4;
							}
						}
						else {
							action_4;
							goto tree_3;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_7: if_finish_condition goto break_7;
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
					goto tree_73;
				}
				else {
					if (condition_c) {
						action_6;
						goto tree_73;
					}
					else {
						action_11;
						goto tree_73;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_i) {
								action_6;
								goto tree_4;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_7;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_7;
							}
							else {
								action_11;
								goto tree_7;
							}
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_8: if_finish_condition goto break_8;
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
								goto tree_73;
							}
							else {
								action_14;
								goto tree_73;
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									action_16;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_12;
											goto tree_4;
										}
										else {
											action_15;
											goto tree_4;
										}
									}
									else {
										action_15;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_h) {
								action_9;
								goto tree_47;
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
							goto tree_82;
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_63;
							}
							else {
								action_6;
								goto tree_61;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
						goto tree_73;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
						else {
							action_4;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								if (condition_h) {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								if (condition_h) {
									action_3;
									goto tree_2;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							if (condition_h) {
								action_3;
								goto tree_61;
							}
							else {
								action_2;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_47: if_finish_condition goto break_47;
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					action_11;
					goto tree_73;
				}
			}
			else {
				if (condition_p) {
					if (condition_k) {
						if (condition_d) {
							if (condition_c) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_12;
							goto tree_4;
						}
					}
					else {
						action_6;
						goto tree_47;
					}
				}
				else {
					action_6;
					goto tree_82;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							action_11;
							goto tree_7;
						}
					}
					else {
						if (condition_k) {
							if (condition_d) {
								if (condition_c) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
				}
				else {
					action_6;
					goto tree_8;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							action_5;
							goto tree_4;
						}
						else {
							action_2;
							goto tree_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_61: if_finish_condition goto break_61;
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								action_12;
								goto tree_4;
							}
							else {
								if (condition_i) {
									action_16;
									goto tree_4;
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								action_11;
								goto tree_3;
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_11;
							goto tree_63;
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_d) {
									action_12;
									goto tree_4;
								}
								else {
									if (condition_i) {
										action_16;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_11;
									goto tree_3;
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_63: if_finish_condition goto break_63;
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_12;
										goto tree_4;
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
									goto tree_3;
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
								goto tree_63;
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_73: if_finish_condition goto break_73;
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
						goto tree_73;
					}
					else {
						if (condition_c) {
							action_6;
							goto tree_73;
						}
						else {
							action_11;
							goto tree_73;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_i) {
									action_6;
									goto tree_4;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_4;
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								action_12;
								goto tree_4;
							}
						}
						else {
							action_6;
							goto tree_47;
						}
					}
					else {
						action_6;
						goto tree_82;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_6;
								goto tree_73;
							}
							else {
								action_11;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
						else {
							action_6;
							goto tree_82;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
							goto tree_73;
						}
						else {
							if (condition_c) {
								action_4;
								goto tree_73;
							}
							else {
								action_7;
								goto tree_73;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									if (condition_d) {
										if (condition_c) {
											action_5;
											goto tree_4;
										}
										else {
											action_8;
											goto tree_4;
										}
									}
									else {
										action_8;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_3;
									goto tree_2;
								}
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_63;
							}
							else {
								action_3;
								goto tree_61;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
								goto tree_7;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_7;
								}
								else {
									action_11;
									goto tree_7;
								}
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_i) {
										action_6;
										goto tree_4;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_4;
										}
										else {
											action_12;
											goto tree_4;
										}
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								action_6;
								goto tree_47;
							}
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
									goto tree_7;
								}
								else {
									if (condition_c) {
										action_6;
										goto tree_7;
									}
									else {
										action_11;
										goto tree_7;
									}
								}
							}
							else {
								if (condition_k) {
									if (condition_d) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_6;
										goto tree_3;
									}
									else {
										action_6;
										goto tree_2;
									}
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
								goto tree_7;
							}
							else {
								if (condition_k) {
									if (condition_i) {
										if (condition_d) {
											action_5;
											goto tree_4;
										}
										else {
											action_10;
											goto tree_4;
										}
									}
									else {
										action_5;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										action_4;
										goto tree_3;
									}
									else {
										action_2;
										goto tree_2;
									}
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						if (condition_n) {
							action_6;
							goto tree_8;
						}
						else {
							action_2;
							goto tree_8;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	tree_82: if_finish_condition goto break_82;
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
								goto tree_73;
							}
							else {
								if (condition_c) {
									action_6;
									goto tree_73;
								}
								else {
									action_11;
									goto tree_73;
								}
							}
						}
						else {
							action_11;
							goto tree_73;
						}
					}
					else {
						action_11;
						goto tree_73;
					}
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_d) {
								if (condition_g) {
									if (condition_b) {
										if (condition_i) {
											action_6;
											goto tree_4;
										}
										else {
											if (condition_c) {
												action_6;
												goto tree_4;
											}
											else {
												action_12;
												goto tree_4;
											}
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_12;
											goto tree_4;
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_16;
										goto tree_4;
									}
								}
								else {
									action_12;
									goto tree_4;
								}
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
										goto tree_3;
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_11;
									goto tree_3;
								}
							}
							else {
								action_6;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
									goto tree_63;
								}
								else {
									action_11;
									goto tree_63;
								}
							}
							else {
								action_11;
								goto tree_63;
							}
						}
						else {
							action_6;
							goto tree_61;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
					goto tree_73;
				}
				else {
					if (condition_p) {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
							goto tree_63;
						}
						else {
							action_2;
							goto tree_61;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
										goto tree_7;
									}
									else {
										if (condition_c) {
											action_6;
											goto tree_7;
										}
										else {
											action_11;
											goto tree_7;
										}
									}
								}
								else {
									action_11;
									goto tree_7;
								}
							}
							else {
								action_11;
								goto tree_7;
							}
						}
						else {
							if (condition_k) {
								if (condition_d) {
									if (condition_g) {
										if (condition_b) {
											if (condition_i) {
												action_6;
												goto tree_4;
											}
											else {
												if (condition_c) {
													action_6;
													goto tree_4;
												}
												else {
													action_12;
													goto tree_4;
												}
											}
										}
										else {
											action_12;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
								else {
									if (condition_i) {
										if (condition_g) {
											if (condition_b) {
												action_12;
												goto tree_4;
											}
											else {
												action_16;
												goto tree_4;
											}
										}
										else {
											action_16;
											goto tree_4;
										}
									}
									else {
										action_12;
										goto tree_4;
									}
								}
							}
							else {
								if (condition_i) {
									if (condition_g) {
										if (condition_b) {
											action_6;
											goto tree_3;
										}
										else {
											action_11;
											goto tree_3;
										}
									}
									else {
										action_11;
										goto tree_3;
									}
								}
								else {
									action_6;
									goto tree_2;
								}
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
							goto tree_7;
						}
						else {
							if (condition_k) {
								if (condition_i) {
									if (condition_d) {
										action_5;
										goto tree_4;
									}
									else {
										action_10;
										goto tree_4;
									}
								}
								else {
									action_5;
									goto tree_4;
								}
							}
							else {
								if (condition_i) {
									action_4;
									goto tree_3;
								}
								else {
									action_2;
									goto tree_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
						goto tree_8;
					}
					else {
						action_2;
						goto tree_8;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
						goto tree_7;
					}
					else {
						if (condition_k) {
							if (condition_i) {
								if (condition_d) {
									action_5;
									goto tree_4;
								}
								else {
									action_10;
									goto tree_4;
								}
							}
							else {
								action_5;
								goto tree_4;
							}
						}
						else {
							if (condition_i) {
								action_4;
								goto tree_3;
							}
							else {
								action_2;
								goto tree_2;
							}
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
						goto tree_1;
					}
					else {
						action_1;
						goto tree_0;
					}
				}
			}
		}
	break_0:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_4;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_4;
						}
						else {
							action_7;
						}
					}
					else {
						action_4;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							action_3;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					action_2;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_1:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_11;
				}
				else {
					if (condition_h) {
						if (condition_c) {
							action_11;
						}
						else {
							action_14;
						}
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_h) {
						action_9;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
				else {
					if (condition_h) {
						action_9;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_2:
		if (condition_o) {
			if (condition_j) {
				action_11;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_11;
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_3:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_b) {
						action_6;
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_4:
		if (condition_o) {
			if (condition_j) {
				action_6;
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						action_6;
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_4;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_7:
		if (condition_o) {
			if (condition_j) {
				if (condition_i) {
					action_6;
				}
				else {
					if (condition_c) {
						action_6;
					}
					else {
						action_11;
					}
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_8:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_i) {
						action_11;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_11;
							}
							else {
								action_14;
							}
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_h) {
							action_9;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_h) {
							action_9;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
				}
			}
			else {
				if (condition_j) {
					if (condition_i) {
						action_4;
					}
					else {
						if (condition_h) {
							if (condition_c) {
								action_4;
							}
							else {
								action_7;
							}
						}
						else {
							action_4;
						}
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_h) {
								action_3;
							}
							else {
								action_2;
							}
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_h) {
								action_3;
							}
							else {
								action_2;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_47:
		if (condition_o) {
			if (condition_j) {
				if (condition_c) {
					if (condition_g) {
						if (condition_b) {
							action_6;
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					action_11;
				}
			}
			else {
				if (condition_p) {
					action_6;
				}
				else {
					action_6;
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_j) {
						if (condition_c) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_6;
					}
				}
				else {
					action_6;
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						action_2;
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_61:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					action_11;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							action_11;
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							action_11;
						}
						else {
							if (condition_i) {
								action_11;
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_63:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_b) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						action_11;
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							if (condition_b) {
								action_6;
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_b) {
								if (condition_i) {
									action_6;
								}
								else {
									if (condition_c) {
										action_6;
									}
									else {
										action_11;
									}
								}
							}
							else {
								action_11;
							}
						}
						else {
							if (condition_i) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_73:
		if (condition_o) {
			if (condition_n) {
				if (condition_j) {
					if (condition_i) {
						action_6;
					}
					else {
						if (condition_c) {
							action_6;
						}
						else {
							action_11;
						}
					}
				}
				else {
					if (condition_p) {
						action_6;
					}
					else {
						action_6;
					}
				}
			}
			else {
				if (condition_r) {
					if (condition_j) {
						if (condition_i) {
							action_6;
						}
						else {
							if (condition_c) {
								action_6;
							}
							else {
								action_11;
							}
						}
					}
					else {
						if (condition_p) {
							action_6;
						}
						else {
							action_6;
						}
					}
				}
				else {
					if (condition_j) {
						if (condition_i) {
							action_4;
						}
						else {
							if (condition_c) {
								action_4;
							}
							else {
								action_7;
							}
						}
					}
					else {
						if (condition_p) {
							if (condition_i) {
								action_4;
							}
							else {
								action_3;
							}
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_3;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_n) {
						if (condition_j) {
							if (condition_i) {
								action_6;
							}
							else {
								if (condition_c) {
									action_6;
								}
								else {
									action_11;
								}
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_r) {
							if (condition_j) {
								if (condition_i) {
									action_6;
								}
								else {
									if (condition_c) {
										action_6;
									}
									else {
										action_11;
									}
								}
							}
							else {
								if (condition_i) {
									action_6;
								}
								else {
									action_6;
								}
							}
						}
						else {
							if (condition_j) {
								action_4;
							}
							else {
								if (condition_i) {
									action_4;
								}
								else {
									action_2;
								}
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						if (condition_n) {
							action_6;
						}
						else {
							action_2;
						}
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	break_82:
		if (condition_o) {
			if (condition_r) {
				if (condition_j) {
					if (condition_g) {
						if (condition_b) {
							if (condition_i) {
								action_6;
							}
							else {
								if (condition_c) {
									action_6;
								}
								else {
									action_11;
								}
							}
						}
						else {
							action_11;
						}
					}
					else {
						action_11;
					}
				}
				else {
					if (condition_p) {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
					else {
						if (condition_i) {
							if (condition_g) {
								if (condition_b) {
									action_6;
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							action_6;
						}
					}
				}
			}
			else {
				if (condition_j) {
					action_4;
				}
				else {
					if (condition_p) {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
			}
		}
		else {
			if (condition_s) {
				if (condition_p) {
					if (condition_r) {
						if (condition_j) {
							if (condition_g) {
								if (condition_b) {
									if (condition_i) {
										action_6;
									}
									else {
										if (condition_c) {
											action_6;
										}
										else {
											action_11;
										}
									}
								}
								else {
									action_11;
								}
							}
							else {
								action_11;
							}
						}
						else {
							if (condition_i) {
								if (condition_g) {
									if (condition_b) {
										action_6;
									}
									else {
										action_11;
									}
								}
								else {
									action_11;
								}
							}
							else {
								action_6;
							}
						}
					}
					else {
						if (condition_j) {
							action_4;
						}
						else {
							if (condition_i) {
								action_4;
							}
							else {
								action_2;
							}
						}
					}
				}
				else {
					if (condition_r) {
						action_6;
					}
					else {
						action_2;
					}
				}
			}
			else {
				if (condition_p) {
					if (condition_j) {
						action_4;
					}
					else {
						if (condition_i) {
							action_4;
						}
						else {
							action_2;
						}
					}
				}
				else {
					if (condition_t) {
						action_2;
					}
					else {
						action_1;
					}
				}
			}
		}
		continue;
	}

}

int Spaghetti1(const cv::Mat1b &img, cv::Mat1i &imgLabels) {
	
    imgLabels = cv::Mat1i(img.size());
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;
	//Tree of labels
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;

    firstScanSpaghetti1(img, imgLabels, P, lunique);

	uint nLabel = flattenL(P, lunique);

	// Second scan
	if (imgLabels.rows & 1){
		if (imgLabels.cols & 1){
			//Case 1: both rows and cols odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (c + 1<imgLabels.cols) {
							if (img_row[c + 1]>0)
								imgLabels_row[c + 1] = iLabel;
							else
								imgLabels_row[c + 1] = 0;
							if (r + 1<imgLabels.rows) {
								if (img_row_fol[c]>0)
									imgLabels_row_fol[c] = iLabel;
								else
									imgLabels_row_fol[c] = 0;
								if (img_row_fol[c + 1]>0)
									imgLabels_row_fol[c + 1] = iLabel;
								else
									imgLabels_row_fol[c + 1] = 0;
							}
						}
						else if (r + 1<imgLabels.rows) {
							if (img_row_fol[c]>0)
								imgLabels_row_fol[c] = iLabel;
							else
								imgLabels_row_fol[c] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (c + 1<imgLabels.cols) {
							imgLabels_row[c + 1] = 0;
							if (r + 1<imgLabels.rows) {
								imgLabels_row_fol[c] = 0;
								imgLabels_row_fol[c + 1] = 0;
							}
						}
						else if (r + 1<imgLabels.rows) {
							imgLabels_row_fol[c] = 0;
						}
					}
				}
			}
		}//END Case 1
		else{
			//Case 2: only rows odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row[c + 1]>0)
							imgLabels_row[c + 1] = iLabel;
						else
							imgLabels_row[c + 1] = 0;
						if (r + 1<imgLabels.rows) {
							if (img_row_fol[c]>0)
								imgLabels_row_fol[c] = iLabel;
							else
								imgLabels_row_fol[c] = 0;
							if (img_row_fol[c + 1]>0)
								imgLabels_row_fol[c + 1] = iLabel;
							else
								imgLabels_row_fol[c + 1] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row[c + 1] = 0;
						if (r + 1<imgLabels.rows) {
							imgLabels_row_fol[c] = 0;
							imgLabels_row_fol[c + 1] = 0;
						}
					}
				}
			}
		}// END Case 2
	} 
	else{
		if (imgLabels.cols & 1){
			//Case 3: only cols odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row_fol[c]>0)
							imgLabels_row_fol[c] = iLabel;
						else
							imgLabels_row_fol[c] = 0;
						if (c + 1<imgLabels.cols) {
							if (img_row[c + 1]>0)
								imgLabels_row[c + 1] = iLabel;
							else
								imgLabels_row[c + 1] = 0;
							if (img_row_fol[c + 1]>0)
								imgLabels_row_fol[c + 1] = iLabel;
							else
								imgLabels_row_fol[c + 1] = 0;
						}
					}
					else{
						imgLabels_row[c] = 0;
						imgLabels_row_fol[c] = 0;
						if (c + 1<imgLabels.cols) {
							imgLabels_row[c + 1] = 0;
							imgLabels_row_fol[c + 1] = 0;
						}
					}
				}
			}
		}// END case 3
		else{
			//Case 4: nothing odd
			for (int r = 0; r < imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row[c + 1] > 0)
							imgLabels_row[c + 1] = iLabel;
						else
							imgLabels_row[c + 1] = 0;
						if (img_row_fol[c] > 0)
							imgLabels_row_fol[c] = iLabel;
						else
							imgLabels_row_fol[c] = 0;
						if (img_row_fol[c + 1] > 0)
							imgLabels_row_fol[c + 1] = iLabel;
						else
							imgLabels_row_fol[c + 1] = 0;
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row[c + 1] = 0;
						imgLabels_row_fol[c] = 0;
						imgLabels_row_fol[c + 1] = 0;
					}
				}
			}
		}//END case 4
	}

	fastFree(P);
	return nLabel;
}
