import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  // ILayoutRestorer
} from '@jupyterlab/application';

// import {readFileSync} from 'fs';
// const in_path = '../examples/8510393.ipynb';
// console.log(readFileSync(in_path, 'utf8'));

import {
  ICommandPalette,
  CommandToolbarButton//,
  // MainAreaWidget,
  // WidgetTracker
} from '@jupyterlab/apputils';

import { ILauncher } from '@jupyterlab/launcher';

import { IMainMenu } from '@jupyterlab/mainmenu';

import { ITranslator } from '@jupyterlab/translation';

import { Menu } from '@lumino/widgets';

import { find } from '@lumino/algorithm';

import { CommandRegistry } from '@lumino/commands';

import { Token } from '@lumino/coreutils';

import { ExamplePanel } from './panel';
// import {KernelSpyModel} from './model';

import {
  IDisposable, DisposableDelegate
} from '@lumino/disposable';

import {
  DocumentRegistry
} from '@jupyterlab/docregistry';

import {
  NotebookPanel, INotebookModel//, INotebookTracker
} from '@jupyterlab/notebook';



// import { AttachedProperty } from '@lumino/properties';

// import { KernelSpyView } from './widget';

/**
 * The command IDs used by the console plugin.
 */
namespace CommandIDs {
  export const create = 'kernel-messaging:create';
  export const newSpy = 'kernelspy:new';
}


/**
 * IDs of the commands added by this extension.
 */
// import {parse} from "../node_modules/@andrewhead/python-program-analysis"
// import {NAME, CALL, DOT, ASSIGN} from '@andrewhead/python-program-analysis'

// import {
//   WalkListener, MagicsRewriter, RefSet, walk,
//   parse
//   ControlFlowGraph,
//   printNode, DataflowAnalyzer, DataflowAnalyzerOptions, slice, SliceDirection,
//   LocationSet, SyntaxNode, Location
// } from '@andrewhead/python-program-analysis'


/**
 * The token identifying the JupyterLab plugin.
 */
export const IKernelSpyExtension = new Token<IKernelSpyExtension>(
  'jupyter.extensions.kernelspy'
);

export type IKernelSpyExtension = DocumentRegistry.IWidgetExtension<
  NotebookPanel,
  INotebookModel
>;

// const spyProp = new AttachedProperty<KernelSpyView, string>({
//   create: () => '',
//   name: 'SpyTarget'
// });

/**
 * Initialization data for the extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'kernel-messaging',
  autoStart: true,
  optional: [ILauncher/*,ICommandPalette, IMainMenu, ILayoutRestorer*/],
  requires: [ICommandPalette, IMainMenu, ITranslator/*, INotebookTracker*/],
  provides: IKernelSpyExtension,
  activate: activate
};

/**
 * Activate the JupyterLab extension.
 *
 * @param app Jupyter Front End
 * @param palette Jupyter Commands Palette
 * @param mainMenu Jupyter Menu
 * @param translator Jupyter Translator
 * @param launcher [optional] Jupyter Launcher
 */
function activate(
  app: JupyterFrontEnd,
  // tracker: INotebookTracker,
  palette: ICommandPalette,
  mainMenu: IMainMenu,
  translator: ITranslator,
  launcher: ILauncher | null

  // tracker: INotebookTracker,
  // restorer: ILayoutRestorer | null
): void {
  const manager = app.serviceManager;
  const { commands, shell } = app;
  const category = 'Extension Examples';
  const trans = translator.load('jupyterlab');
  console.log(app);

  // Add launcher
  if (launcher) {
    launcher.add({
      command: CommandIDs.create,
      category: category
    });
  }


  var commEx = new NBCommExtension();
  app.docRegistry.addWidgetExtension('Notebook', commEx);
  /**
   * Creates a example panel.
   *
   * @returns The panel
   */
   // after this panel is created, commEx starts receiving messages without
   //  errors
  async function createPanel(): Promise<ExamplePanel> {
    const panel = new ExamplePanel(manager, translator);
    shell.add(panel, 'main');
    console.log(panel._sessionContext.sessionManager);


    // console.log(commEx);
    // console.log(commEx.the_panel);
    commEx.the_panel = panel;
    commEx.the_panel._example._notebook_panel = commEx.nb;
    // console.log(commEx.the_panel);
    // console.log(panel._notebook_sessionContext);

    // start to find kernel of current notebook?
    // var a_kernel_spy_model = new KernelSpyModel(panel._notebook_model._sessionContext.session?.kernel);
    // console.log(a_kernel_spy_model);
    return panel;
  }

  // add menu tab
  const exampleMenu = new Menu({ commands });
  exampleMenu.title.label = trans.__('EDAssistant');
  mainMenu.addMenu(exampleMenu);

  // add commands to registry
  commands.addCommand(CommandIDs.create, {
    label: trans.__('EDAssistant View'),
    caption: trans.__('EDAssistant View'),
    execute: createPanel
  });

  // add items in command palette and menu
  palette.addItem({ command: CommandIDs.create, category });
  exampleMenu.addItem({ command: CommandIDs.create });

}




export
class NBCommExtension implements DocumentRegistry.IWidgetExtension<NotebookPanel, INotebookModel> {
  public the_panel: ExamplePanel;
  public nb: NotebookPanel;


  get getPanel() {
    return this.the_panel;
  }
  /**
   * Create a new extension object.
   */
  createNew(panel: NotebookPanel, context: DocumentRegistry.IContext<INotebookModel>): IDisposable {

	console.log("createNew debug ");
    Promise.all([panel.revealed, panel.sessionContext.ready, context.ready]).then(() => {
		console.log("Promise all debug");
		//const session = context.sessionContext.session;
		const session = panel.sessionContext.session;
		const kernelInstance = session.kernel;
    this.nb = panel;
    console.log(this.nb);
    console.log(this.nb.model);
    console.log(this.nb.model.cells);
    console.log(this.nb.model.cells.get(0).value);
    console.log(context);

 		try {
			console.log("try registerCommTarget ");

			kernelInstance.registerCommTarget('my_comm_target', (comm: any, msg: any) => {
				console.log("registerCommTarget debug");
				// comm is the frontend comm instance
				// msg is the comm_open message, which can carry data

				// Register handlers for later messages:
				//comm.on_msg(function(msg) {console.log("message received: ",msg);});
				//comm.on_close(function(msg) {console.log("Comm my_comm_target closed");});

				comm.onMsg = (msg: any) => {
          console.log("message received: ",msg);
          console.log(msg.content.data);
          console.log(msg.content.data['foo']);
          console.log(msg.content.data['dataframe']);
          comm.send({'foo': 3});

          // not sure if typescript switch can handle this
          if (msg.content.data['dataframe'] != null) {
            this.the_panel._example._df = msg.content.data['dataframe'];
            this.the_panel._example.registerDf(msg.content.data['dataframe']);
          }

        };
				comm.onClose = (msg: any) => {
          console.log("comm onClose");
        };
				comm.send({'foo': 0});
			});
		}
		catch(err) {
			console.log("registerCommTarget error : ",err.message);
		}

    });

    return new DisposableDelegate(() => {
    });

  }




}

///////////////////////////////


export class KernelSpyExtension implements IKernelSpyExtension {
  /**
   *
   */
  constructor(commands: CommandRegistry) {
    this.commands = commands;
  }

  /**
   * Create a new extension object.
   */
  createNew(
    nb: NotebookPanel,
    context: DocumentRegistry.IContext<INotebookModel>
  ): IDisposable {
    // Add buttons to toolbar
    const buttons: CommandToolbarButton[] = [];
    let insertionPoint = -1;
    find(nb.toolbar.children(), (tbb, index) => {
      if (tbb.hasClass('jp-Notebook-toolbarCellType')) {
        insertionPoint = index;
        return true;
      }
      return false;
    });
    console.log('test notebook panel');
    console.log(nb);
    let i = 1;
    for (const id of [CommandIDs.newSpy]) {
      const button = new CommandToolbarButton({ id, commands: this.commands });
      button.addClass('jp-kernelspy-nbtoolbarbutton');
      if (insertionPoint >= 0) {
        nb.toolbar.insertItem(
          insertionPoint + i++,
          this.commands.label(id),
          button
        );
      } else {
        nb.toolbar.addItem(this.commands.label(id), button);
      }
      buttons.push(button);
    }

    return new DisposableDelegate(() => {
      // Cleanup extension here
      for (const btn of buttons) {
        btn.dispose();
      }
    });
  }

  protected commands: CommandRegistry;
}


export default extension;
// export extension1;
