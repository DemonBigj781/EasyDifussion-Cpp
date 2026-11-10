(() => {
    "use strict";
    const modes = [['draw','Draw'],['inpaint','Inpaint'],['advanced','Advanced Inpaint']];
    class ImagePreviewTabs {
        constructor(parent, prefix) {
            if (!document.getElementById('image-preview-tabs-style')) {
                const style=document.createElement('style');style.id='image-preview-tabs-style';
                style.textContent=`.image-preview-tabs{width:280px;max-width:100%;min-width:0}.image-preview-tabs [role=tablist]{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}.image-preview-tabs [role=tab]{min-width:0;padding:6px;white-space:normal;overflow-wrap:anywhere;font:inherit;font-size:.8em;cursor:pointer}.image-preview-tabs [aria-selected=true]{outline:2px solid #6b9edb}.image-preview-tabs [role=tabpanel]{margin-top:8px;text-align:center}.image-preview-tabs [role=tabpanel][hidden]{display:none}.image-preview-tabs img{display:block;width:100%;height:220px;max-width:100%;margin:auto;object-fit:contain}.image-preview-tabs img[hidden]{display:none}.image-preview-tabs small{display:block;line-height:1.4;margin:5px 0;overflow-wrap:anywhere}`;
                document.head.appendChild(style);
            }
            this.root=document.createElement('div');this.root.className='image-preview-tabs';
            const tabs=document.createElement('div');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Image input previews');
            this.root.appendChild(tabs);this.buttons={};this.panels={};this.images={};
            for (const [mode,label] of modes) {
                const button=document.createElement('button');button.type='button';button.id=`${prefix}-tab-${mode}`;
                button.textContent=label;button.setAttribute('role','tab');button.setAttribute('aria-controls',`${prefix}-panel-${mode}`);
                const panel=document.createElement('div');panel.id=`${prefix}-panel-${mode}`;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',button.id);
                const image=document.createElement('img');image.alt=`${label} preview`;image.hidden=true;panel.appendChild(image);
                const caption=document.createElement('small');caption.textContent=mode==='advanced'?'Color-weighted guidance overlay — not painted output.':mode==='inpaint'?'Ordinary inpaint mask overlay.':'Draw image preview.';panel.appendChild(caption);
                button.addEventListener('click',()=>this.select(mode));
                button.addEventListener('keydown',event=>{
                    const index=modes.findIndex(([name])=>name===mode);
                    const next=event.key==='ArrowRight'?(index+1)%3:event.key==='ArrowLeft'?(index+2)%3:event.key==='Home'?0:event.key==='End'?2:null;
                    if(next===null)return;event.preventDefault();this.select(modes[next][0]);this.buttons[modes[next][0]].focus();
                });
                this.buttons[mode]=button;this.panels[mode]=panel;this.images[mode]=image;
                tabs.appendChild(button);this.root.appendChild(panel);
            }
            const note=document.createElement('small');note.textContent='Preview tabs are view-only. Save or apply an editor to change generation inputs.';this.root.appendChild(note);
            parent.appendChild(this.root);this.select('draw');
        }
        select(mode) {
            if (!this.panels[mode]) return;
            for (const [name] of modes) {
                this.panels[name].hidden=name!==mode;this.buttons[name].tabIndex=name===mode?0:-1;
                this.buttons[name].setAttribute('aria-selected',String(name===mode));
            }
        }
        setImages(images) {
            for (const [mode] of modes) {
                const value=images?.[mode];this.images[mode].hidden=!value;
                if(value)this.images[mode].src=value;else this.images[mode].removeAttribute('src');
            }
        }
        static render(base, layer, mode) {
            const width=base.naturalWidth || base.width,height=base.naturalHeight || base.height;
            const scale=Math.min(1,512/Math.max(width,height));
            const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(width*scale));canvas.height=Math.max(1,Math.round(height*scale));
            const ctx=canvas.getContext('2d');ctx.drawImage(base,0,0,canvas.width,canvas.height);
            if(layer) {
                if(mode==='inpaint') {
                    const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;
                    const mask=overlay.getContext('2d');mask.drawImage(layer,0,0,canvas.width,canvas.height);
                    const data=mask.getImageData(0,0,canvas.width,canvas.height);
                    for(let i=0;i<data.data.length;i+=4){data.data[i+3]=Math.round(data.data[i+3]*Math.max(data.data[i],data.data[i+1],data.data[i+2])/255*.6);data.data[i]=255;data.data[i+1]=64;data.data[i+2]=160;}
                    mask.putImageData(data,0,0);ctx.drawImage(overlay,0,0);
                } else ctx.drawImage(layer,0,0,canvas.width,canvas.height);
            }
            return canvas.toDataURL('image/png');
        }
    }
    window.ImagePreviewTabs=ImagePreviewTabs;
})();
