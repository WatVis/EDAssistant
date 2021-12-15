// import { Notebook } from "@jupyterlab/notebook";
// import { Kernel } from "@jupyterlab/services";
// import { JSONArray } from '@phosphor/coreutils';
// import { Cell, CodeCell, CodeCellModel } from '@jupyterlab/cells';
//
// export function sendJupyterCodeCells(
//   kernelInstance: Kernel.IKernelConnection,
//   notebook: Notebook,
//   filter: string
// ): void {
//   const comm = kernelInstance.connectToComm('smarteda.getcodecells');
//   const codeCells = <JSONArray>getCodeCellsByTag(notebook, filter)
//     .map((cell: CodeCell): object => ({
//         cell_type: cell.model.type,
//         ...cell.model.toJSON()
//       })
//     );
//
//   comm.open();
//   comm.send({ code_cells: codeCells });
//   comm.dispose();
// }
// 
// export function getCodeCellsByTag(notebook: Notebook, tag: string): Cell[] {
//   let cells = notebook.widgets || [];
//
//   return cells.filter((cell) => {
//     const tags: any = cell.model.metadata.get('tags');
//
//     return (
//       cell.model instanceof CodeCellModel &&
//       tags && tags.length && tags.includes(tag)
//     );
//   });
// }
