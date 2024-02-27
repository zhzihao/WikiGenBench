prompt_vanilla= \
    'Based on your own knowledge, you are assigned to write a Wikipedia article on the topic.\
    Organize the content of your article by sections. Before writing each section, always starts with "==SECTION NAME==".'

prompt_vanilla_given_outline= \
    'Based on your own knowledge, you are assigned to write a Wikipedia article on the topic.\
    Follow the given outline to Organize the content of your article by sections. Before writing each section, always starts with "==SECTION NAME==".'

prompt_writer = \
    'Based on the given information, you are assigned to write a Wikipedia article on the topic.\
    Organize the content of your article by sections. Before writing each section, always starts with "==SECTION NAME==".\
    You must cite the most relevant document for every sentence you write, in the format of "This is an example sentence.[k]", where k denotes Document k.'

prompt_writer_given_outline = \
    'Based on the given information, you are assigned to write a Wikipedia article on the topic.\
    Follow the given outline to organize the content of your article by sections. Before writing each section, always starts with "==SECTION NAME==".\
    You must cite the most relevant document for every sentence you write, in the format of "This is an example sentence.[k]", where k denotes Document k.'

prompt_writer_per_section = \
    'Based on the given information, you are assigned to write a Wikipedia article on the topic.\
    Only write the specified section. Before writing the section, starts with "==SECTION NAME==".\
    You must cite the most relevant document for every sentence you write, in the format of "This is an example sentence.[k]", where k denotes Document k.'

prompt_outline = \
    'Based on the given information, you are assigned to make an outline for the Wikipedia article on the topic.\
    List 5~10 sections in the outline, in the format of "==SECTION NAME==".'

prompt_rewrite = \
    "Please rewrite the passage below to enhance its fluency and coherence while maintaining the original meaning. \
    Pay attention to the citation indexes [k] at the end of each sentence; these should remain unchanged if the sentence conveys a similar idea to the original. \
    Ensuring that the section headings \"==HEADING==\" remain unchanged. Preserve the length of the passage as much as possible."