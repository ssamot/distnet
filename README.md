# distnet
Distance Memory Recurrent Networks, an memory network like learning scheme
Results Comparison Thus far
<table>
<tbody>
<tr class="odd">
<td align="left"><strong>Task Number/Name | </strong>LSTM | <strong>MemNN-1 | </strong>MemNN-2 | <strong>MemNN-3 | </strong>DMRNN-1| <strong>DMRNN-2 | </strong>DMRNN-3</td>
</tr>
<tr class="even">
<td align="left">QA1 - Single Supporting Fact | 0.5 | 0.992 | 1 | 1 | 1 | 1 | 1</td>
</tr>
<tr class="odd">
<td align="left">QA2 - Two Supporting Facts | 0.2 | 0.38 | 0.844 | 0.886 | 0.362 | 0.812 | 0.966</td>
</tr>
<tr class="even">
<td align="left">QA3 - Three Supporting Facts | 0.2 | 0.231 | 0.684 | 0.781 | 0.346 | 0.707 | 0.748</td>
</tr>
<tr class="odd">
<td align="left">QA4 - Two Arg. Relations | 0.61 | 0.772 | 0.978 | 0.866 | 0.891 | 0.885 | 0.891</td>
</tr>
<tr class="even">
<td align="left">QA5 - Three Arg. Relations | 0.7 | 0.89 | 0.866 | 0.856 | 0.847 | 0.838 | 0.834</td>
</tr>
<tr class="odd">
<td align="left">QA6 - Yes/No Questions | 0.48 | 0.928 | 0.977 | 0.972 | 0.999 | 1 | 1</td>
</tr>
<tr class="even">
<td align="left">QA7 - Counting | 0.49 | 0.841 | 0.746 | 0.817 | 0.881 | 0.767 | 0.862</td>
</tr>
<tr class="odd">
<td align="left">QA8 - Lists/Sets | 0.45 | 0.868 | 0.883 | 0.907 | 0.95 | 0.969 | 0.965</td>
</tr>
<tr class="even">
<td align="left">QA9 - Simple Negation | 0.64 | 0.949 | 0.98 | 0.981 | 1 | 1 | 1</td>
</tr>
<tr class="odd">
<td align="left">QA10 - Indefinite Knowledge | 0.44 | 0.894 | 0.95 | 0.935 | 0.997 | 0.999 | 0.999</td>
</tr>
<tr class="even">
<td align="left">QA11 - Basic Coreference | 0.62 | 0.916 | 0.988 | 0.997 | 0.926 | 0.928 | 0.955</td>
</tr>
<tr class="odd">
<td align="left">QA12 - Conjunction | 0.74 | 0.996 | 1 | 0.999 | 1 | 1 | 1</td>
</tr>
<tr class="even">
<td align="left">QA13 - Compound Coreference | 0.94 | 0.937 | 0.998 | 0.998 | 0.946 | 0.954 | 0.969</td>
</tr>
<tr class="odd">
<td align="left">QA14 - Time Reasoning | 0.27 | 0.631 | 0.919 | 0.931 | 0.784 | 1 | 1</td>
</tr>
<tr class="even">
<td align="left">QA15 - Basic Deduction | 0.21 | 0.536 | 0.995 | 1 | 0.839 | 1 | 1</td>
</tr>
<tr class="odd">
<td align="left">QA16 - Basic Induction | 0.23 | 0.526 | 0.487 | 0.973 | 0.467 | 0.457 | 0.457</td>
</tr>
<tr class="even">
<td align="left">QA17 - Positional Reasoning | 0.51 | 0.556 | 0.588 | 0.596 | 0.56 | 0.522 | 0.544</td>
</tr>
<tr class="odd">
<td align="left">QA18 - Size Reasoning | 0.52 | 0.904 | 0.897 | 0.906 | 0.914 | 0.907 | 0.913</td>
</tr>
<tr class="even">
<td align="left">QA19 - Path Finding | 0.08 | 0.093 | 0.101 | 0.12 | 0.133 | 0.109 | 0.141</td>
</tr>
<tr class="odd">
<td align="left">QA20 - Agent’s Motivations | 0.91 | 1 | 0.999 | 1 | 1 | 1 | 1</td>
</tr>
<tr class="even">
<td align="left">Overall Mean | 0.487 | 0.742 | 0.844 | 0.87605 | 0.7921 | 0.8427 | 0.8622</td>
</tr>
<tr class="odd">
<td align="left">0.95 Threshold | 0 | 3 | 9 | 9 | 6 | 10 | 12</td>
</tr>
</tbody>
</table>

