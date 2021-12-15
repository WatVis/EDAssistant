import {
  SessionContext,
  ISessionContext//,
  //sessionContextDialogs
} from '@jupyterlab/apputils';

// import { parse as python3Parse, parser } from './python3';

import {
  ITranslator,
  nullTranslator,
  TranslationBundle
} from '@jupyterlab/translation';

import { ServiceManager } from '@jupyterlab/services';

import { Message } from '@lumino/messaging';

import { StackedPanel } from '@lumino/widgets';

import { KernelView } from './widget';

import {
  KernelModel//,
  // KernelSpyModel
} from './model';

// import {readFileSync} from 'fs';
// import * as path from 'path';
// const fs = require("fs");
// import * as file from 'url-loader!./back_end.py'

/**
 * The class name added to the panels.
 */
const PANEL_CLASS = 'jp-RovaPanel';

// function delay(ms: number) {
//     return new Promise( resolve => setTimeout(resolve, ms) );
// }

/**
 * A panel which has the ability to add other children.
 */
export class ExamplePanel extends StackedPanel {

  public _model: KernelModel;
  public _sessionContext: SessionContext;
  public _example: KernelView;
  public _notebook_sessionContext: SessionContext;
  public _notebook_model: KernelModel;
  // public _notebook_spy_model: KernelSpyModel;

  private _translator: ITranslator;
  private _trans: TranslationBundle;



  constructor(manager: ServiceManager.IManager, translator?: ITranslator) {
    super();
    this._translator = translator || nullTranslator;
    this._trans = this._translator.load('jupyterlab');
    this.addClass(PANEL_CLASS);
    this.id = 'SmartEDA';
    this.title.label = this._trans.__('SmartEDA View');
    this.title.closable = true;

    this._sessionContext = new SessionContext({
      sessionManager: manager.sessions,
      specsManager: manager.kernelspecs,
      name: 'Backend kernel'
    });

    // this._notebook_sessionContext = new SessionContext({
    //   sessionManager: manager.sessions,
    //   specsManager: manager.kernelspecs,
    //   name: 'Corresponded notebook kernel'
    // });

    // this._notebook_model = new KernelModel(this._notebook_sessionContext);


    // this._model = new KernelModel(this._sessionContext);
    // this._example = new KernelView(this._model);//, new KernelSpyModel(null));
    this._example = new KernelView();
    this.addWidget(this._example);



    void this._sessionContext
      .initialize()
      .then(async value => {
        if (value) {
          // await sessionContextDialogs.selectKernel(this._sessionContext);
          // this._model.read_backend_file();
      }
    }).catch(reason => {
        console.error(
          `Failed to initialize the session in ExamplePanel.\n${reason}`
        );
      });

    // void this._notebook_sessionContext
    //   .initialize()
    //   .then(async value => {
    //     if (value) {
    //       await sessionContextDialogs.selectKernel(this._notebook_sessionContext);
    //       this._notebook_spy_model= new KernelSpyModel(this._notebook_model._sessionContext.session?.kernel);
    //       // console.log(this._notebook_spy_model);
    //       this._example = new KernelView(this._model, this._notebook_spy_model);
    //
    //       // this.addWidget(this._example);
    //
    //   }
    // }).catch(reason => {
    //     console.error(
    //       `Failed to initialize the session in ExamplePanel.\n${reason}`
    //     );
    //   });

    // console.log(parser('import numpy as np\n'));

  }


  get session(): ISessionContext {
    return this._sessionContext;
  }

  dispose(): void {
    this._sessionContext.dispose();
    super.dispose();
  }

  protected onCloseRequest(msg: Message): void {
    super.onCloseRequest(msg);
    this.dispose();
  }


}





//
