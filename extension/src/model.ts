import { ISessionContext } from '@jupyterlab/apputils';

import { IOutput } from '@jupyterlab/nbformat';

import { Kernel, KernelMessage } from '@jupyterlab/services';

import { ISignal, Signal } from '@lumino/signaling';

import { EventEmitter } from '@angular/core';



// import { IIterator } from '@lumino/algorithm';

// import { VDomModel } from '@jupyterlab/apputils';

// no kernelview, just kernelmodel
// import { Kernel, KernelMessage } from '@jupyterlab/services';

export class KernelModel {
  constructor(session: ISessionContext) {
    this._sessionContext = session;
  }

  get future(): Kernel.IFuture<
    KernelMessage.IExecuteRequestMsg,
    KernelMessage.IExecuteReplyMsg
  > | null {
    return this._future;
  }

  set future(
    value: Kernel.IFuture<
      KernelMessage.IExecuteRequestMsg,
      KernelMessage.IExecuteReplyMsg
    > | null
  ) {
    this._future = value;
    if (!value) {
      return;
    }
    value.onIOPub = this._onIOPub;
  }

  get output(): IOutput | null {
    return this._output;
  }

  get stateChanged(): ISignal<KernelModel, void> {
    return this._stateChanged;
  }

  execute(code: string): void {
    if (!this._sessionContext || !this._sessionContext.session?.kernel) {
      return;
    }

    this.future = this._sessionContext.session?.kernel?.requestExecute({
      code
    });
  }

  public read_backend_file() {
    this.execute("f = open('src/back_end.py', 'r')\nprint(f.read())");
    this._initiate_backend = true;
  }

  private _onIOPub = (msg: KernelMessage.IIOPubMessage): void => {
    const msgType = msg.header.msg_type;
    console.log(msg);
    switch (msgType) {
      case 'execute_result':

      case 'display_data':

      case 'update_display_data':
        this._output = msg.content as IOutput;
        this._stateChanged.emit();
        break;
      case 'stream':
        // two cases:
        // 1. printed backend code, in this case, we need to ask kernel to run it.
        // 2. widget.tsx 's buttons just did something... it printed.
        var the_msg_content = msg.content as any;

        if (this._initiate_backend) {
          // we only initiate for once.
          this._initiate_backend = false;
          // the_msg_content.text contains our backend code.
          // so let the backend kernel run it.
          this.execute(the_msg_content.text);
        } else {
          this._back_end_result.push(the_msg_content.text);

          console.log('complete emited.');
          this.complete.emit(the_msg_content.text); // this fires an event
          this._stateChanged.emit();
        }
        break;
      default:
        break;
    }
    return;
  };

  private _future: Kernel.IFuture<
    KernelMessage.IExecuteRequestMsg,
    KernelMessage.IExecuteReplyMsg
  > | null = null;
  private _output: IOutput | null = null;
  public _sessionContext: ISessionContext;
  private _stateChanged = new Signal<KernelModel, void>(this);
  public _back_end_result: string[] = [];
  private _initiate_backend: boolean = false;

  public complete = new EventEmitter();
  public _recommend_panel: boolean = false;
  public _dna_panel: boolean = false;
}


// /////////////////////////////////////////////////////////////////////////////
// export type MessageThread = {
//   args: Kernel.IAnyMessageArgs;
//   children: MessageThread[];
// };
//
// function isHeader(
//   candidate: { [key: string]: undefined } | KernelMessage.IHeader
// ): candidate is KernelMessage.IHeader {
//   return candidate.msg_id !== undefined;
// }
//
// export class ThreadIterator implements IIterator<ThreadIterator.IElement> {
//   constructor(threads: MessageThread[], collapsed: { [key: string]: boolean }) {
//     this._threads = threads;
//     this._collapsed = collapsed;
//     this._index = -1;
//     this._child = null;
//   }
//
//   iter() {
//     return this;
//   }
//
//   next(): ThreadIterator.IElement | undefined {
//     if (this._child) {
//       const next = this._child.next();
//       if (next !== undefined) {
//         return next;
//       }
//       this._child = null;
//     }
//     // Move to next thread
//     ++this._index;
//     if (this._index >= this._threads.length) {
//       return undefined;
//     }
//     const entry = this._threads[this._index];
//     if (
//       entry.children.length > 0 &&
//       !this._collapsed[entry.args.msg.header.msg_id]
//     ) {
//       // Iterate over children after this
//       this._child = new ThreadIterator(entry.children, this._collapsed);
//     }
//     return { args: entry.args, hasChildren: entry.children.length > 0 };
//   }
//
//   clone(): ThreadIterator {
//     const r = new ThreadIterator(this._threads, this._collapsed);
//     r._index = this._index;
//     if (this._child) {
//       r._child = this._child.clone();
//     }
//     return r;
//   }
//
//   private _index: number;
//   private _child: ThreadIterator | null;
//
//   private _threads: MessageThread[];
//   private _collapsed: { [key: string]: boolean };
// }
//
// export namespace ThreadIterator {
//   export interface IElement {
//     args: Kernel.IAnyMessageArgs;
//     hasChildren: boolean;
//   }
// }



/**
 * Model for a kernel spy.
 */
// export class KernelSpyModel extends VDomModel {
//   constructor(kernel?: Kernel.IKernelConnection | null) {
//     super();
//     this.kernel = kernel ?? null;
//   }
//
//   clear() {
//     this._log.splice(0, this._log.length);
//     this._messages = {};
//     this._childLUT = {};
//     this._roots = [];
//     this.stateChanged.emit(void 0);
//   }
//
//   get kernel() {
//     return this._kernel;
//   }
//
//   set kernel(value: Kernel.IKernelConnection | null) {
//     if (this._kernel) {
//       this._kernel.anyMessage.disconnect(this.onMessage, this);
//     }
//     this._kernel = value;
//     if (this._kernel) {
//       this._kernel.anyMessage.connect(this.onMessage, this);
//     }
//     console.log(this._kernel);
//   }
//
//   get log(): ReadonlyArray<Kernel.IAnyMessageArgs> {
//     return this._log;
//   }
//
//   get tree(): MessageThread[] {
//     return this._roots.map(rootId => {
//       return this.getThread(rootId, false);
//     });
//   }
//
//   depth(args: Kernel.IAnyMessageArgs | null): number {
//     if (args === null) {
//       return -1;
//     }
//     let depth = 0;
//     while ((args = this._findParent(args))) {
//       ++depth;
//     }
//     return depth;
//   }
//
//   getThread(msgId: string, ancestors = true): MessageThread {
//     const args = this._messages[msgId];
//     if (ancestors) {
//       // Work up to root, then work downwards
//       let root = args;
//       let candidate;
//       while ((candidate = this._findParent(root))) {
//         root = candidate;
//       }
//       return this.getThread(root.msg.header.msg_id, false);
//     }
//
//     const childMessages = this._childLUT[msgId] || [];
//     const childThreads = childMessages.map(childId => {
//       return this.getThread(childId, false);
//     });
//     const thread: MessageThread = {
//       args: this._messages[msgId],
//       children: childThreads
//     };
//     return thread;
//   }
//
//   protected onMessage(
//     sender: Kernel.IKernelConnection,
//     args: Kernel.IAnyMessageArgs
//   ) {
//     const { msg } = args;
//     this._log.push(args);
//     this._messages[msg.header.msg_id] = args;
//     const parent = this._findParent(args);
//     if (parent === null) {
//       this._roots.push(msg.header.msg_id);
//     } else {
//       const header = parent.msg.header;
//       this._childLUT[header.msg_id] = this._childLUT[header.msg_id] || [];
//       this._childLUT[header.msg_id].push(msg.header.msg_id);
//     }
//
//     var the_content = msg.content as any;
//     if (the_content.code != null) {
//       console.log('User runned a code cell. Here is the code: ');
//       console.log(the_content.code);
//       this._all_runned_code.push(the_content.code);
//     }
//     this.stateChanged.emit();
//   }
//
//   private _findParent(
//     args: Kernel.IAnyMessageArgs
//   ): Kernel.IAnyMessageArgs | null {
//     if (isHeader(args.msg.parent_header)) {
//       return this._messages[args.msg.parent_header.msg_id] || null;
//     }
//     return null;
//   }
//
//   public _log: Kernel.IAnyMessageArgs[] = [];
//
//   private _kernel: Kernel.IKernelConnection | null = null;
//
//   public _all_runned_code: string[] = [];
//   public _messages: { [key: string]: Kernel.IAnyMessageArgs } = {};
//   public _childLUT: { [key: string]: string[] } = {};
//   private _roots: string[] = [];
// }
