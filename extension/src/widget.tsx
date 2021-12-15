import {
  ReactWidget//,
  // UseSignal
} from '@jupyterlab/apputils';
// import { parse } from "@andrewhead/python-program-analysis";
// import {readFileSync} from 'fs';
// const fs = require('@types/node/fs');
// console.log(fs);
// const in_path = '../examples/8510393.ipynb';
import hljs from 'highlight.js';
// hljs.initHighlightingOnLoad();
// import * as React from 'react';
import * as _d3 from "d3";
import {
  NotebookPanel//, INotebookModel//, INotebookTracker
} from '@jupyterlab/notebook';

// import { KernelModel } from './model';
// 'use strict';

// Yizhi's parser...
// import { Notebook } from '@ervin/python-program-analysis';
// import { SliceDirection } from '@ervin/python-program-analysis';
// import * as ast from '@ervin/python-program-analysis/dist/es5/python-parser';

import * as React from 'react';
import axios from 'axios';
// import {
//   KernelSpyModel//,
//   // ThreadIterator
// } from './model';

import '../style/index.css';

// glocal variables
var _current_dem_dna_seq: number;    // used by pager.
var _notebook_panel_detail_visible: boolean = false; // used by pager, false then 'previous'
                                                      // has no impact.
var _dna_sequences_len: number = 0;   // number of dna sequences we currently have.

const max_seq_per_page: number = 5;

var _curr_selected_group:any = 'nothing';

var _curr_selected_line_relative:any = 'nothing';

var _curr_dna_page: number = 0;
var _dna_triggered: boolean = false;
var _dna_pages: number = 0;

export class KernelView extends ReactWidget {


  constructor() {//, spy_model:KernelSpyModel) {
    super();
    // this._model = model;
    // this._spy_model = spy_model;
    // add observe function...
    // this._model.complete.subscribe((msg)=>{
    //   // we update it to false here. o/w there will be some logical bugs.
    //   _notebook_panel_detail_visible = false;
    //   this.update_rec_panel(msg);
    //   this.update_dna_panel(msg);
    // })
  }

  get getDf() {
    return this._df;
  }

  public registerDf(a_df: Array<Array<any>>) {
    // ._df has been set before this function call.
    // this._df = a_df;
    console.log('df is ready.');
    console.log(a_df);
  }


  private update_rec_panel(predicted_cells:any) {
    // msg is an array with length 5. array of strings(operations).
    console.log('updating rec_panel');
    console.log(this._notebook_panel.model.cells);
    console.log(predicted_cells);

    // clear
    (document.getElementById("Recommend_display") as HTMLInputElement).innerHTML = "";
    //
    var font_size = ["12px", "12px", "12px", "12px", "12px"];
    // var tag_color = ["#FFA880", "#FFBA99", "#FFCBB3", "#FFDCCC", "#FFEEE5"];
    var tag_color = ['#707B7C', '#7F8C8D', '#99A3A4', '#B2BABB', '#CCD1D1'];

    // our recommendation panel.
    var recmd_panel = _d3.select('#Recommend_display');

    for (let i = 0; i < predicted_cells.length; i++) {
      if (i > 4) {
        console.log('Recommended operations are > 5.');
        break;
      }
      console.log(predicted_cells[i]);
      recmd_panel.append("span")
                 .attr('class', "chip")
                 .style('font-size', font_size[i])
                 .style('background-color', tag_color[i])
                 .text(predicted_cells[i])
                 .on("mouseover", function(event, d) {
                   _d3.select(this)
                      .style("border", "1px solid");
                 })
                 .on("mouseout", function(event, d) {
                   _d3.select(this)
                      .style("border", "");
                 });
    }
  }

  private pagination(paths_mid:any[][], operations_mid:any[][], break_blocks_mid:any[][][]) {

    const dna_seq_interval = 70;      // interval between 2 dna sequences
    const dna_seq_width = 20;         // width of dna sequence width.

    var paths:any[][][] = [];
    var operations:any[][][][] = [];
    var break_blocks:any[][][] = [];

    for (var i = 0; i < paths_mid.length; i++) {
      var page_path:any[][] = [];
      var page_break_blocks: any[][] = [];
      var page_operations: any[][][] = [];

      for (var j = 0; j < max_seq_per_page; j++) {
        if (i + j == paths_mid.length) {
          break;
        }
        // console.log(i + j, i + j, paths_mid[i + j]);
        page_path.push(paths_mid[i + j]);
        page_break_blocks.push(break_blocks_mid[i + j]);
        page_operations.push(operations_mid[i + j]);
      }
      i = i + max_seq_per_page - 1;

      paths.push(page_path);
      break_blocks.push(page_break_blocks);
      operations.push(page_operations);
    }


    //// move everything to right.
    // move path
    for (let i = 1; i < paths.length; i++) {
      // console.log(i + 1, 'th page');
      // console.log(i * (10 + 20) * max_seq_per_page);
      for (let j = 0; j < paths[i].length; j++) {
        // move paths to the right.
        for (let k = 0; k < paths[i][j].length; k++) {
          // normally paths[i][j].length == 2. except the
          //    scenario we have folding...

          // 150 => (dna_seq_interval + dna_seq_width) * max_seq_per_page = 20;
          paths[i][j][k].left = paths[i][j][k].left - i * (dna_seq_width + dna_seq_interval) * max_seq_per_page;
          paths[i][j][k].right = paths[i][j][k].right - i * (dna_seq_width + dna_seq_interval) * max_seq_per_page;

        }
        // also do this to operations.
        for (let m = 0; m < operations[i][j].length; m++) {
          // 150 => (dna_seq_interval + dna_seq_width) * max_seq_per_page = 20;
          // for operation's top
          operations[i][j][m][0].left = operations[i][j][m][0].left - i * (dna_seq_width + dna_seq_interval) * max_seq_per_page;
          operations[i][j][m][0].right = operations[i][j][m][0].right - i * (dna_seq_width + dna_seq_interval) * max_seq_per_page;
          // for operation's bottom
          operations[i][j][m][1].left = operations[i][j][m][1].left - i * (dna_seq_width + dna_seq_interval) * max_seq_per_page;
          operations[i][j][m][1].right = operations[i][j][m][1].right - i * (dna_seq_width + dna_seq_interval) * max_seq_per_page;
        }


      }
    }

    // we do not paginate keywords' position, just use converted index.
    for (let ntbk = 0; ntbk < this._curr_dna_keywords_posi.length; ntbk++) {
      var curr_page_index = Math.floor(ntbk / max_seq_per_page);
      if (curr_page_index == 0) {
        continue;
      }
      // else, decreament
      for (let kwd = 0; kwd < this._curr_dna_keywords_posi[ntbk].length; kwd++) {
        this._curr_dna_keywords_posi[ntbk][kwd].x = this._curr_dna_keywords_posi[ntbk][kwd].x - curr_page_index * (dna_seq_width + dna_seq_interval) * max_seq_per_page;
      }
    }



    return [paths, operations, break_blocks];
  }

  private render_curr_dna_page(paths:any[][], operations_posi:any[][][], break_blocks:any[][], msg:any, _curr_dna_ops:any, _belongs:any, _group_info_no_detail:any, _group_info:any) {
    const colors = ['#EC7063', '#A569BD', '#5DADE2', '#58D68D', '#F4D03F'];

    const dna_keywords = this._curr_dna_keywords;
    const dna_keywords_posi = this._curr_dna_keywords_posi;


    // clear
    (document.getElementById("sequence_layer") as HTMLInputElement).innerHTML = "";
    (document.getElementById("operation_layer") as HTMLInputElement).innerHTML = "";
    (document.getElementById("keyword_layer") as HTMLInputElement).innerHTML = "";

    // find the div
    var div = _d3.select("#tooltip")
            .style("opacity", 0);

    // set the area generator
    var areaGenerator = _d3.area();
    areaGenerator.x0(function (d:any) {
      return d.left;
    }).x1(function (d:any) {
      return d.right;
    }).y(function (d:any) {
      return d.y;
    });
    areaGenerator
      .defined(function (d) {
        return d !== null;
      });
    // generate dna sequences
    for (let j = 0; j < paths.length; j++) {
      // generate path
      var a_path = areaGenerator(paths[j]);
      _d3.select('#sequence_layer')
          .append('path')
          .attr('d', a_path)
          .attr('class', 'dna_eda_sequence')
          .attr('id', 'dna_eda_seq' + String(j))
          .on('mouseover', function(event, d) {
            div.transition()
                .duration(200)
                .style("opacity", .9);
            var idx_before_pagination = _curr_dna_page * max_seq_per_page + j;
            var curr_code = "";
            for (var counter = 0; counter < _curr_dna_ops[idx_before_pagination].length; counter++) {
              if (counter == 5) {
                break;
              }
              curr_code = curr_code + _curr_dna_ops[idx_before_pagination][counter] + '<br/>';
            }
            div.html(curr_code)
            // div	.html('operation 1 count: ' + String(break_blocks[j].filter(v => v === 1).length) + "<br/>"
            //           + 'operation 2 count: ' + String(break_blocks[j].filter(v => v === 2).length) + "<br/>"
            //           + 'operation 3 count: ' + String(break_blocks[j].filter(v => v === 3).length) + "<br/>"
            //           + 'operation 4 count: ' + String(break_blocks[j].filter(v => v === 4).length))
                .style("left", (event.pageX) + "px")
                .style("top", (event.pageY - 28) + "px");
          })
          .on('mouseout', function(d) {
            div.transition()
                .duration(500)
                .style("opacity", 0);
          })
          .on('click', function(d) {
            // set border to red
            _d3.selectAll('.dna_eda_sequence')
                .style('stroke', 'black');

            _d3.select('#dna_eda_seq' + String(j))
                .style('stroke', 'red');

            // set the curr selected group to nothing, since we clicked a path...
            _curr_selected_group = 'nothing';

            // clear the notebook panel
            (document.getElementById("notebook_display") as HTMLInputElement).innerText = "";
            // (document.getElementById("arrow") as HTMLInputElement).innerText = "";
            var notebook_panel = _d3.select('#notebook_display');
            // click will set it to be visible.
            _notebook_panel_detail_visible = true;
            // the current clicked notebook.
            _current_dem_dna_seq = j;
            const index_before_pagination = _curr_dna_page * max_seq_per_page + _current_dem_dna_seq;
            var current_notebook_panel_cell_counter = 0;
            for (var r = 0; r < _belongs[index_before_pagination].length; r++) {
              const curr_belong = _belongs[index_before_pagination][r];
              if (curr_belong == 0) {
                // increment the counter?

              } else {
                const start = r;
                var end = start;

                for (var s = r; s < _belongs[index_before_pagination].length; s++) {
                  if (_belongs[index_before_pagination][s] == 1) {
                    // r = r + 1;
                    if (s == (_belongs[index_before_pagination].length - 1)) {
                      // end. auto-stop
                      r = s;
                    }
                    continue;
                  } else if (s == _belongs[index_before_pagination].length - 1) {
                    // end, but 0, it will auto-stop
                    r = s - 1;
                  } else {
                    // encountered 0, time to stop...
                    r = s - 1; // later check posi s again.
                    break;
                  }
                }
                end = r;

                // This block of code was originally used for generating a code cell.
                // var curr_code_group = "";
                // // from start to end (inclusive), is a 'group' of continuous code from current block.
                // for (var s = start; s <= end; s++) {
                //   console.log(s);
                //   curr_code_group = curr_code_group + _curr_dna_ops[index_before_pagination][s];
                // }
                // // this is the code for current entire code group.
                // console.log(curr_code_group);

                const first_block_top = 10;

                var a_container = notebook_panel.append('div')
                      .attr('class', 'the_containers')
                      .attr('id', 'container' + current_notebook_panel_cell_counter.toString())
                      // .style('top', (first_block_top).toString() + 'px');
                      .style('top', (current_notebook_panel_cell_counter * (5) + first_block_top).toString() + 'px');


                var one_notebook_block = a_container.append('div')
                      .attr('class', 'fake_textarea notebook_panel_code')
                      .attr('id', 'notebook_panel_code' + current_notebook_panel_cell_counter.toString())
                      .attr('disabled', 'true');
                      // .append('pre')  // we don't need syntax highlighting for now
                      // .style('height', '100%')
                      // .style('width', '100%')
                      // .append('code') // we don't need syntax highlighting for now
                      // .style('height', '100%')
                      // .style('width', '100%')
                      // .style('font-size', '11px')
                      // .style('background-color', '#F5F5F5')
                      // .attr('id', 'notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString()) // we don't need syntax highlighting for now
                      // .attr('class', 'python hljs'); // we don't need syntax highlighting for now

                for (var s = start; s <= end; s++) {
                  one_notebook_block.append('pre')
                                    .style('margin', '0 0 0 0')
                                    .style('width', '100%')
                                    .append('code')
                                    // .style('height', '100%')
                                    .style('width', '100%')
                                    .style('font-size', '11px')
                                    .style('background-color', '#F5F5F5')
                                    .attr('id', 'notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString() + '$' + (s - start).toString())
                                    .attr('class', 'python hljs'); // we don't need syntax highlighting for now
                                    (document.getElementById('notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString() + '$' + (s - start).toString()) as HTMLInputElement).innerHTML =
                                                hljs.highlight('python', _curr_dna_ops[index_before_pagination][s], true).value;
                }

                // one_notebook_block;
                // (document.getElementById('notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString()) as HTMLInputElement).innerHTML =
                //             hljs.highlight('python', curr_code_group, true).value;
                            // curr_code_group;

                current_notebook_panel_cell_counter = current_notebook_panel_cell_counter + 1;
              }
            }
          });
      //  generate keywords.
      const idx_before_pagination = _curr_dna_page * max_seq_per_page + j;
      var kwd_layer = _d3.select('#keyword_layer');
      for (let m = 0; m < dna_keywords[idx_before_pagination].length; m++) {
        kwd_layer.append('text')
                 .attr('class', 'keywords')
                 .attr("x", dna_keywords_posi[idx_before_pagination][m].x)
                 .attr("y", dna_keywords_posi[idx_before_pagination][m].y)
                 .text(dna_keywords[idx_before_pagination][m]);

      }
    }

    // set the width and height. or the overflow: scroll will not work
    (document.getElementById('DNA_svg') as HTMLInputElement).style.width =
      ((document.getElementById('drawing') as HTMLInputElement).getBoundingClientRect().width + 30) + 'px';
    (document.getElementById('DNA_svg') as HTMLInputElement).style.height =
      ((document.getElementById('drawing') as HTMLInputElement).getBoundingClientRect().height + 15) + 'px';


    // render dna operations
    for (let h = 0; h < operations_posi.length; h++) {
      const index_before_pagination = _curr_dna_page * max_seq_per_page + h;
      for (let m = 0; m < operations_posi[h].length; m++) {
        // h-th notebook, m-th operation.
        var one_path = areaGenerator(operations_posi[h][m]);


        var the_operation = _d3.select('#operation_layer')
                                  .append('path')
                                  .attr('d', one_path)
                                  .attr('id', 'operation' + String(h) + String(m))
                                  .style('fill', function (d, i) {
                                    return colors[break_blocks[h][m]]; //colors[h][break_blocks[h][m]];
                                  }).style('opacity', 1);
        the_operation.on('click', function(d) {
          // pass the click event to paths, show the cells in notebook panel first.
          _d3.select('#dna_eda_seq' + String(h))
             .dispatch('click');

          // scroll to proper position
          const curr_group = _group_info_no_detail[index_before_pagination][m];
          console.log('resultsss');
          console.log(curr_group);
          _d3.select('#container' + String(curr_group))
            .style('background', 'red');
          _d3.select('#notebook_panel_code' + String(curr_group))
            .style('width', '99%');


          // we make that line of code obvious (a single line)
          // find relative position...
          var curr_pointer = m;
          while (curr_pointer >= 0) {
            if (_group_info_no_detail[index_before_pagination][curr_pointer] != curr_group) {
              curr_pointer += 1;
              break;
            }
            curr_pointer -= 1;
          }
          if (curr_pointer == -1) {
            curr_pointer = 0;
          }
          const relative_posi = m - curr_pointer;
          var myCode = document.getElementById("notebook_panel_highlighted_code" + String(curr_group) + "$" + relative_posi.toString());
          myCode.style.backgroundColor = '#F3D6C0';

          // we scroll to that position (line)
          var myElement = document.getElementById('container' + String(curr_group));
          // var myElement = document.getElementById("notebook_panel_highlighted_code" + String(curr_group) + "$" + relative_posi.toString());
          // console.log('offsetTop');
          // console.log(myElement.offsetTop - 20);
          document.getElementById('notebook_display').scrollTop = myElement.offsetTop - 20;

          // set the curr selected group
          _curr_selected_group = _group_info[index_before_pagination][m];
          _curr_selected_line_relative = relative_posi;

          console.log('check group info');
          console.log(_group_info);
          console.log(_group_info_no_detail)

        });
      }
    }
  }

  private show_all_notebooks(show_details: boolean, current_notebook_after_pagination: number) {
    // set border to red
    _d3.selectAll('.dna_eda_sequence')
        .style('stroke', 'black');

    _d3.select('#dna_eda_seq' + String(current_notebook_after_pagination))
        .style('stroke', 'red');

    // clear the notebook panel
    (document.getElementById("notebook_display") as HTMLInputElement).innerText = "";
    // (document.getElementById("arrow") as HTMLInputElement).innerText = "";
    var notebook_panel = _d3.select('#notebook_display');
    // click will set it to be visible.
    _notebook_panel_detail_visible = true;
    // the current clicked notebook.
    _current_dem_dna_seq = current_notebook_after_pagination;
    const index_before_pagination = _curr_dna_page * max_seq_per_page + _current_dem_dna_seq;
    var current_notebook_panel_cell_counter = 0;
    for (var r = 0; r < this._belongs[index_before_pagination].length; r++) {
      const curr_belong = this._belongs[index_before_pagination][r];

      const start = r;
      var end = start;

      for (var s = r; s < this._belongs[index_before_pagination].length; s++) {
        if (this._belongs[index_before_pagination][s] == curr_belong) {
          // r = r + 1;
          if (s == (this._belongs[index_before_pagination].length - 1)) {
            // end. auto-stop
            r = s;
          }
          continue;
        } else if (s == this._belongs[index_before_pagination].length - 1) {
          // end, but 0, it will auto-stop
          r = s - 1;
        } else {
          // encountered 0, time to stop...
          r = s - 1; // later check posi s again.
          break;
        }
      }
      end = r;


      // var curr_code_group = "";
      // // from start to end (inclusive), is a 'group' of continuous code from current block.
      // for (var s = start; s <= end; s++) {
      //   console.log(s);
      //   curr_code_group = curr_code_group + this._curr_dna_ops[index_before_pagination][s];
      // }

      // this is the code for current entire code group.

      const first_block_top = 10;

      var a_container = notebook_panel.append('div')
            .attr('class', 'the_containers')
            .attr('id', 'container' + current_notebook_panel_cell_counter.toString())
            .style('top', (current_notebook_panel_cell_counter * (5) + first_block_top).toString() + 'px');
      if (curr_belong == 1) {
        a_container.style('background', 'orange');
      }

      if (current_notebook_panel_cell_counter == _curr_selected_group) {
        a_container.style('background', 'red');
      }


      var one_notebook_block = a_container.append('div')
            .attr('class', 'fake_textarea notebook_panel_code')
            .attr('id', 'notebook_panel_code' + current_notebook_panel_cell_counter.toString())
            .attr('disabled', 'true');
            // .append('pre')  // we need syntax highlighting
            // .style('height', '100%')
            // .style('width', '100%')
            // .append('code')
            // .style('height', '100%')
            // .style('width', '100%')
            // .style('font-size', '11px')
            // .style('background-color', '#F5F5F5')
            // .attr('id', 'notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString()) // we don't need syntax highlighting for now
            // .attr('class', 'python hljs');
      // one_notebook_block;

      for (var s = start; s <= end; s++) {

        one_notebook_block.append('pre')  // we need syntax highlighting
                          .style('width', '100%')
                          .style('margin', '0 0 0 0')
                          .append('code')
                          .style('width', '100%')
                          .style('font-size', '11px')
                          .style('background-color', '#F5F5F5')
                          .attr('id', 'notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString() + '$' + (s - start).toString()) // we don't need syntax highlighting for now
                          .attr('class', 'python hljs');
        (document.getElementById('notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString() + '$' + (s - start).toString()) as HTMLInputElement).innerHTML =
                          hljs.highlight('python', this._curr_dna_ops[index_before_pagination][s], true).value;
      }

      if (curr_belong == 1 || current_notebook_panel_cell_counter == _curr_selected_group) {
        _d3.select('#notebook_panel_code'  + current_notebook_panel_cell_counter.toString())
            .style('width', '99%');
      }
      // (document.getElementById('notebook_panel_highlighted_code' + current_notebook_panel_cell_counter.toString()) as HTMLInputElement).innerHTML =
      //             hljs.highlight('python', curr_code_group, true).value;

      current_notebook_panel_cell_counter = current_notebook_panel_cell_counter + 1;
    }
    // scroll to a proper position
    if (_curr_selected_group != 'nothing') {
      var myElement = document.getElementById('container' + String(_curr_selected_group));
      document.getElementById('notebook_display').scrollTop = myElement.offsetTop - 20;
      console.log('offsetTop');
      console.log(myElement.offsetTop - 20);

      // highlight the line...
      var myCode = document.getElementById("notebook_panel_highlighted_code" + String(_curr_selected_group) + "$" + _curr_selected_line_relative.toString());
      myCode.style.backgroundColor = '#F3D6C0';
    }
  }


  private update_dna_panel(returned:any) {
    if (true) {
      // clear the original dna panel
      (document.getElementById("sequence_layer") as HTMLInputElement).innerHTML = "";
      (document.getElementById("operation_layer") as HTMLInputElement).innerHTML = "";
      (document.getElementById("keyword_layer") as HTMLInputElement).innerHTML = "";
      // remove anything inside notebook_display
      (document.getElementById("notebook_display") as HTMLInputElement).innerHTML = "";


      // var split_returned:string[] = returned.split('$', 4);
      // console.log(split_returned);
      // 2D array of operation types
      var msg = returned['all_operation_types']//JSON.parse(split_returned[0]);
      // 2D array of strings (code)
      this._curr_dna_ops = returned['all_operations']//JSON.parse(split_returned[1].replace(/@/g, '\''));
      // 3D array of 1 or 0
      var if_belongs_to = returned['all_if_demonstrated']//JSON.parse(split_returned[2]);
      // 2D array of strings
      this._curr_dna_keywords = returned['all_kwds']//JSON.parse(split_returned[3]);
      console.log(this._curr_dna_keywords);

      this._belongs = if_belongs_to;


      this._curr_dna_msg = msg;
      // this._model._dna_panel = false;
      // console.log('dna_panel triggered');
      // console.log(msg);
      // console.log(this._curr_dna_ops);
      // console.log(if_belongs_to);

      // use d3 to update the svg.
      // data we need:
      var [path, operation, break_blocks_mid_state] = this.process_dna_input_array(msg, if_belongs_to);

      // const dna_panel_height = (document.getElementById("DNA_panel") as HTMLInputElement).getBoundingClientRect().height;
      // console.log("dna svg height is " + dna_panel_height);
      // const dna_seq_stopping = +(dna_panel_height) - 90;
      // var [paths_mid_state, operations_posi_mid_state] = this.handle_folding(path, operation, dna_seq_stopping); // 300 is the max len
      var [paths, operations_posi, break_blocks] = this.pagination(path, operation, break_blocks_mid_state);




      this._paths = paths;
      this._operations_posi = operations_posi;
      this._break_blocks = break_blocks;
      // console.log(this._operations_posi);


      _curr_dna_page = 0;
      _dna_triggered = true;
      _dna_pages = paths.length;

      // update number of dna sequences we have
      _dna_sequences_len = paths[0].length;
      this.render_curr_dna_page(paths[0], operations_posi[0], break_blocks[0], msg, this._curr_dna_ops, this._belongs, this._group_info_no_detail, this._group_info);
    }
  }

  private process_dna_input_array(msg:any, if_belongs_to:any) {

    // define some constants
    const dna_top_adjust = 10;        // dna sequence is ..px down from the top edge
                                      //   of its nearest positioned ancestor:
    const dna_left_adjust = 20;       // dna sequence is ..px right from the left edge

    const dna_seq_interval = 70;      // interval between 2 dna sequences
    const dna_seq_width = 20;         // width of dna sequence width.

    // const dna_block_interval = 20;    // interval between 2 EDA blocks
    const dna_first_last_block_interval = 10; // space of top and bottom
    const dna_operation_interval = 0; // interval between 2 EDA operations
    // const max_len = 200;              // max len we don't need folding
    const dna_operation_height = 4;   // height of each EDA operation
    const dna_keywords_dist_from_seq = 2; // distance between keyword and its sequence
    const dna_keywords_y_shift = 2; // move all keywords down a little bit
    const folding_jump_y = 5;

    var paths:any[][] = [];
    var operations_posi:any[][][] = [];
    this._curr_dna_keywords_posi = [];
    this._group_info_no_detail = [];
    this._group_info = [];

    console.log('entered process_dna_input_array');

    for (var i = 0; i < msg.length; i++) {
      // msg[i] is a notebook
      var single_path = [];
      var single_operation_posi = [];
      var single_dna_keywords_posi = [];
      var single_dna_group_info_no_detail = [];
      var single_dna_group_info = [];
      // TODO: remember to determine if notebook is too long
      var curr_y_posi = dna_first_last_block_interval + dna_top_adjust;

      const curr_left_posi = dna_left_adjust + i * (dna_seq_width + dna_seq_interval);
      const curr_right_posi = dna_left_adjust + i * (dna_seq_width + dna_seq_interval) + dna_seq_width;

      // starting point of path:
      single_path.push({'y': dna_top_adjust,
                        'left': curr_left_posi,
                        'right': curr_right_posi});
      var curr_group_counter_no_detail = 0;
      var curr_group_counter = 0;
      // see each demonstrated operation belongs to which group...
      for (var l = 0; l < msg[i].length; l++) {
        if (if_belongs_to[i][l] == 1) {
          single_dna_group_info_no_detail.push(curr_group_counter_no_detail);
          single_dna_group_info.push(curr_group_counter);
        } else {
          // it is 0 that increment the group counter.
          if (l == 0) {
            curr_group_counter = curr_group_counter + 1;
          } else if (if_belongs_to[i][l - 1] == 1) {
            // this is a new 0
            curr_group_counter_no_detail = curr_group_counter_no_detail + 1;
            curr_group_counter = curr_group_counter + 2;
          }
        }
      }
      this._group_info_no_detail.push(single_dna_group_info_no_detail);
      this._group_info.push(single_dna_group_info);

      for (var j = 0; j < msg[i].length; j++) {
        // msg[i][j] is an operation
        if (if_belongs_to[i][j] == 0) {
          // might need to insert folding here.
          var curr_zero = 0;

          for (var k = j; k < msg[i].length; k++) {
            if (if_belongs_to[i][k] == 0) {
              // update j, so that we actually skip the 0s
              j = j + 1;
              curr_zero = curr_zero + 1
              continue;
            } else if (k == (msg[i].length - 1)) {
              // a 1. but at the end.
              j = k - 1;
              break;
            } else {
              // a 1, not the end. stop
              j = k - 1;
              break;
            }
          }
          if (curr_zero > 4) {    // originally 2



            single_path.push({'y': curr_y_posi + folding_jump_y * 1,
                           'left': curr_left_posi,
                           'right': curr_right_posi});
            single_path.push({'y': curr_y_posi + folding_jump_y * 2,
                           'left':  curr_left_posi + dna_seq_interval * (1 / 16),
                           'right': curr_right_posi + dna_seq_interval * (1 / 16)});
            single_path.push({'y': curr_y_posi + folding_jump_y * 3,
                           'left':  curr_left_posi - dna_seq_interval * (1 / 16),
                           'right': curr_right_posi - dna_seq_interval * (1 / 16)});
            single_path.push({'y': curr_y_posi + folding_jump_y * 4,
                           'left':  curr_left_posi,
                           'right': curr_right_posi});

            curr_y_posi = curr_y_posi + folding_jump_y * 4 + dna_operation_interval;

          } else {
            // no folding, no need to update path.
            // // if this is a big gap at the end... or start
            // if (gap_to_start_end == false) {
            curr_y_posi = curr_y_posi + curr_zero * (dna_operation_height + dna_operation_interval);
            // }
          }
          continue;
        }
        // current is 1.
        if (j == 0) {
          // the first 1.
          single_dna_keywords_posi.push({'x': curr_right_posi + dna_keywords_dist_from_seq,
                                         'y': curr_y_posi + dna_operation_height + dna_keywords_y_shift});
        } else if (if_belongs_to[i][j - 1] == 0) {
          // only add new to these 2 scenarios
          single_dna_keywords_posi.push({'x': curr_right_posi + dna_keywords_dist_from_seq,
                                         'y': curr_y_posi + dna_operation_height + dna_keywords_y_shift});
        }
        single_operation_posi.push([{'y': curr_y_posi,
                                     'left':  curr_left_posi,
                                     'right': curr_right_posi},
                                    {'y': curr_y_posi + dna_operation_height,
                                     'left':  curr_left_posi,
                                     'right': curr_right_posi}]);
        // update starting position
        curr_y_posi = curr_y_posi + dna_operation_height + dna_operation_interval;
      }
      // the ending position of current path.
      single_path.push({'y': curr_y_posi + dna_first_last_block_interval,
                        'left':  curr_left_posi,
                        'right': curr_right_posi});
      paths.push(single_path);
      operations_posi.push(single_operation_posi);
      this._curr_dna_keywords_posi.push(single_dna_keywords_posi);
    }
    // console.log(paths);
    // console.log(operations_posi);
    // console.log(msg);
    // console.log(this._curr_dna_keywords_posi);
    return [paths, operations_posi, msg];
  }



  protected render(): React.ReactElement<any> {
    return (
      <React.Fragment>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"/>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.0/styles/tomorrow.min.css"/>
        <script type="text/javascript"
          src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.0/highlight.min.js"></script>
        <script>hljs.initHighlightingOnLoad();</script>
        <div id="DNA_panel">
          <div id="TOP_search_bar">
            <i
              className="fa fa-search"
              id="Recommend_button"
              onClick={(): void => {
                // Recommond functionality is based on current code.
                axios.post('http://129.97.7.121:5000/find_related', {
                  notebook: JSON.stringify(this._notebook_panel.model.toJSON())//text_area_text
                })
                .then((response:any) => {
                  console.log(response);
                  // var res = response.json();
                  console.log(response.data);
                  this.update_dna_panel(response.data);
                })
                .catch(function (error) {
                  console.log(error);
                });
              }}>
              Search by current cell
            </i>
          </div>
          <div id="svg_overflow">
            <svg id="DNA_svg">
              <g id="drawing">
                <g id="sequence_layer">
                </g>
                <g id="operation_layer">
                </g>
                <g id="keyword_layer">
                </g>
              </g>
            </svg>
          </div>
          <div id="tooltip">
          </div>
          <div id="DNA_pager">
            <button
              id="dna_previous"
              onClick={(): void => {
                if (_dna_triggered) {
                  if (_curr_dna_page == 0) {
                    // no previous
                  } else {
                    _curr_dna_page = _curr_dna_page - 1;
                    this.render_curr_dna_page(this._paths[_curr_dna_page],
                                              this._operations_posi[_curr_dna_page],
                                              this._break_blocks[_curr_dna_page],
                                              this._curr_dna_msg,
                                              this._curr_dna_ops,
                                              this._belongs,
                                              this._group_info_no_detail,
                                              this._group_info);
                    // freeze the click of notebook_panel for a while...
                    _notebook_panel_detail_visible = false;
                    _current_dem_dna_seq = 0;
                    _dna_sequences_len = this._paths[_curr_dna_page].length;
                    (document.getElementById('notebook_display') as HTMLInputElement).innerHTML = "";
                  }
                }
                // else we do nothing
              }}>
              &#8249;
            </button>
            <button
              id="dna_next"
              onClick={(): void => {
                // check if notebook_panel is empty.
                if (_dna_triggered) {
                  if (_curr_dna_page + 1 == _dna_pages) {
                    // no next
                  } else {
                    // move to the right by 1
                    _curr_dna_page = _curr_dna_page + 1;
                    this.render_curr_dna_page(this._paths[_curr_dna_page],
                                              this._operations_posi[_curr_dna_page],
                                              this._break_blocks[_curr_dna_page],
                                              this._curr_dna_msg,
                                              this._curr_dna_ops,
                                              this._belongs,
                                              this._group_info_no_detail,
                                              this._group_info);
                    // freeze the click of notebook_panel for a while...
                    _notebook_panel_detail_visible = false;
                    _current_dem_dna_seq = 0;
                    _dna_sequences_len = this._paths[_curr_dna_page].length;
                    (document.getElementById('notebook_display') as HTMLInputElement).innerHTML = "";
                  }
                }
                // else we do nothing
              }}>
              &#8250;
            </button>
          </div>
        </div>

        <div id="Notebook_panel">
          <div id="notebook_detail_container">
            <i
              className="fa fa-bars"
              id="notebook_details"
              onClick={(): void => {
                console.log('Going to demonstrate entire notebook.');
                this.show_all_notebooks(true, _current_dem_dna_seq);
              }}>
            Details
            </i>
          </div>
          <div id="notebook_display">
          </div>
          <div id="notebook_button_container">
            <button
              id="notebook_previous"
              onClick={(): void => {
                console.log(_current_dem_dna_seq);
                // check if notebook_panel is empty.
                if (_notebook_panel_detail_visible) {
                  if (_current_dem_dna_seq == 0) {
                    // no previous
                  } else {
                    // move to the left by 1
                    _current_dem_dna_seq = _current_dem_dna_seq - 1;
                    // trigger click event, assume non-empty notebook.
                    _d3.select('#dna_eda_seq' + String(_current_dem_dna_seq))
                        .dispatch('click');
                  }
                }
                // else we do nothing
              }}>
              &#8249;
            </button>
            <button
              id="notebook_next"
              onClick={(): void => {
                // check if notebook_panel is empty.
                console.log(_current_dem_dna_seq);
                if (_notebook_panel_detail_visible) {
                  if (_current_dem_dna_seq + 1 >= _dna_sequences_len) {
                    // no next
                  } else {
                    // move to the right by 1
                    _current_dem_dna_seq = _current_dem_dna_seq + 1;
                    // trigger click event, assume non-empty notebook.
                    _d3.select('#dna_eda_seq' + String(_current_dem_dna_seq))
                        .dispatch('click');
                  }
                }
                // else we do nothing
              }}>
              &#8250;
            </button>
          </div>
        </div>


        <div id="Recommend_panel">
          <div id="Recommend_button_container">
            <i
              className="fa fa-search"
              id="Rec_button"
              onClick={(): void => {
                axios.post('http://129.97.7.121:5000/predict_next', {
                  notebook: JSON.stringify(this._notebook_panel.model.toJSON())
                })
                .then((response:any) => {
                  console.log(response);
                  // var res = response.json();
                  console.log(response.data);
                  this.update_rec_panel(response.data.top_operations);
                })
                .catch(function (error) {
                  console.log(error);
                });
              }}>
              Predict next
            </i>
          </div>

          <div id="Recommend_display">
          </div>
        </div>


      </React.Fragment>
    );
  }

  // public
  public _df: any;
  //private _model: KernelModel;

  private _paths: any[][][];
  private _operations_posi: any[][][][];
  private _break_blocks: any[][][];
  private _belongs: any[][];
  private _group_info_no_detail: any[];
  private _group_info: any[];

  private _curr_dna_msg: any;
  private _curr_dna_ops: any;
  private _curr_dna_keywords: any;
  private _curr_dna_keywords_posi: any;
  // private _curr_selected_group:any;

  public _notebook_panel: NotebookPanel; // The LHS notebook. use _notebook_panel.model.cells to find all cell contents.
}
