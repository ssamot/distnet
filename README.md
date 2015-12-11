# distnet
Distance Memory Recurrent Networks, an memory network like learning scheme
Results Comparison Thus far
<table>
<thead>
<tr class="header">
<th align="left"><strong>Task Number/Name</strong></th>
<th align="left"><strong>LSTM</strong></th>
<th align="left"><strong>MemNN-1</strong></th>
<th align="left"><strong>MemNN-2</strong></th>
<th align="left"><strong>MemNN-3</strong></th>
<th align="left"><strong>DMRNN-1</strong></th>
<th align="left"><strong>DMRNN-2</strong></th>
<th align="left"><strong>DMRNN-3</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">QA1 - Single Supporting Fact</td>
<td align="left">0.5</td>
<td align="left">0.992</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="even">
<td align="left">QA2 - Two Supporting Facts</td>
<td align="left">0.2</td>
<td align="left">0.38</td>
<td align="left">0.844</td>
<td align="left">0.886</td>
<td align="left">0.362</td>
<td align="left">0.812</td>
<td align="left">0.966</td>
</tr>
<tr class="odd">
<td align="left">QA3 - Three Supporting Facts</td>
<td align="left">0.2</td>
<td align="left">0.231</td>
<td align="left">0.684</td>
<td align="left">0.781</td>
<td align="left">0.346</td>
<td align="left">0.707</td>
<td align="left">0.748</td>
</tr>
<tr class="even">
<td align="left">QA4 - Two Arg. Relations</td>
<td align="left">0.61</td>
<td align="left">0.772</td>
<td align="left">0.978</td>
<td align="left">0.866</td>
<td align="left">0.891</td>
<td align="left">0.885</td>
<td align="left">0.891</td>
</tr>
<tr class="odd">
<td align="left">QA5 - Three Arg. Relations</td>
<td align="left">0.7</td>
<td align="left">0.89</td>
<td align="left">0.866</td>
<td align="left">0.856</td>
<td align="left">0.847</td>
<td align="left">0.838</td>
<td align="left">0.834</td>
</tr>
<tr class="even">
<td align="left">QA6 - Yes/No Questions</td>
<td align="left">0.48</td>
<td align="left">0.928</td>
<td align="left">0.977</td>
<td align="left">0.972</td>
<td align="left">0.999</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="odd">
<td align="left">QA7 - Counting</td>
<td align="left">0.49</td>
<td align="left">0.841</td>
<td align="left">0.746</td>
<td align="left">0.817</td>
<td align="left">0.881</td>
<td align="left">0.767</td>
<td align="left">0.862</td>
</tr>
<tr class="even">
<td align="left">QA8 - Lists/Sets</td>
<td align="left">0.45</td>
<td align="left">0.868</td>
<td align="left">0.883</td>
<td align="left">0.907</td>
<td align="left">0.95</td>
<td align="left">0.969</td>
<td align="left">0.965</td>
</tr>
<tr class="odd">
<td align="left">QA9 - Simple Negation</td>
<td align="left">0.64</td>
<td align="left">0.949</td>
<td align="left">0.98</td>
<td align="left">0.981</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="even">
<td align="left">QA10 - Indefinite Knowledge</td>
<td align="left">0.44</td>
<td align="left">0.894</td>
<td align="left">0.95</td>
<td align="left">0.935</td>
<td align="left">0.997</td>
<td align="left">0.999</td>
<td align="left">0.999</td>
</tr>
<tr class="odd">
<td align="left">QA11 - Basic Coreference</td>
<td align="left">0.62</td>
<td align="left">0.916</td>
<td align="left">0.988</td>
<td align="left">0.997</td>
<td align="left">0.926</td>
<td align="left">0.928</td>
<td align="left">0.955</td>
</tr>
<tr class="even">
<td align="left">QA12 - Conjunction</td>
<td align="left">0.74</td>
<td align="left">0.996</td>
<td align="left">1</td>
<td align="left">0.999</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="odd">
<td align="left">QA13 - Compound Coreference</td>
<td align="left">0.94</td>
<td align="left">0.937</td>
<td align="left">0.998</td>
<td align="left">0.998</td>
<td align="left">0.946</td>
<td align="left">0.954</td>
<td align="left">0.969</td>
</tr>
<tr class="even">
<td align="left">QA14 - Time Reasoning</td>
<td align="left">0.27</td>
<td align="left">0.631</td>
<td align="left">0.919</td>
<td align="left">0.931</td>
<td align="left">0.784</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="odd">
<td align="left">QA15 - Basic Deduction</td>
<td align="left">0.21</td>
<td align="left">0.536</td>
<td align="left">0.995</td>
<td align="left">1</td>
<td align="left">0.839</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="even">
<td align="left">QA16 - Basic Induction</td>
<td align="left">0.23</td>
<td align="left">0.526</td>
<td align="left">0.487</td>
<td align="left">0.973</td>
<td align="left">0.467</td>
<td align="left">0.457</td>
<td align="left">0.457</td>
</tr>
<tr class="odd">
<td align="left">QA17 - Positional Reasoning</td>
<td align="left">0.51</td>
<td align="left">0.556</td>
<td align="left">0.588</td>
<td align="left">0.596</td>
<td align="left">0.56</td>
<td align="left">0.522</td>
<td align="left">0.544</td>
</tr>
<tr class="even">
<td align="left">QA18 - Size Reasoning</td>
<td align="left">0.52</td>
<td align="left">0.904</td>
<td align="left">0.897</td>
<td align="left">0.906</td>
<td align="left">0.914</td>
<td align="left">0.907</td>
<td align="left">0.913</td>
</tr>
<tr class="odd">
<td align="left">QA19 - Path Finding</td>
<td align="left">0.08</td>
<td align="left">0.093</td>
<td align="left">0.101</td>
<td align="left">0.12</td>
<td align="left">0.133</td>
<td align="left">0.109</td>
<td align="left">0.141</td>
</tr>
<tr class="even">
<td align="left">QA20 - Agent’s Motivations</td>
<td align="left">0.91</td>
<td align="left">1</td>
<td align="left">0.999</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
<td align="left">1</td>
</tr>
<tr class="odd">
<td align="left">Overall Mean</td>
<td align="left">0.487</td>
<td align="left">0.742</td>
<td align="left">0.844</td>
<td align="left">0.87605</td>
<td align="left">0.7921</td>
<td align="left">0.8427</td>
<td align="left">0.8622</td>
</tr>
<tr class="even">
<td align="left">0.95 Threshold</td>
<td align="left">0</td>
<td align="left">3</td>
<td align="left">9</td>
<td align="left">9</td>
<td align="left">6</td>
<td align="left">10</td>
<td align="left">12</td>
</tr>
</tbody>
</table>

          
