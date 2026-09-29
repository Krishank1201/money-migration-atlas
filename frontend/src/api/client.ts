import axios from 'axios';
import {
  InvestigationReport,
  TestCase,
  ModelInfoResponse,
  GNNModelInfoResponse,
  EvidencePackage,
  Chain,
  SahyogRequest,
  VaspRegistryItem,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const api = {
  checkHealth: async (): Promise<{ status: string }> => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  getTestCases: async (): Promise<TestCase[]> => {
    const res = await apiClient.get<TestCase[]>('/api/v1/demo/test-cases');
    return res.data;
  },

  investigate: async (chain: Chain | string, address: string, maxHops: number = 6): Promise<InvestigationReport> => {
    const res = await apiClient.post<InvestigationReport>(
      `/api/v1/agentic/investigate/${chain}/${address}?max_hops=${maxHops}`
    );
    return res.data;
  },

  getEvidenceJson: async (chain: Chain | string, address: string): Promise<EvidencePackage> => {
    const res = await apiClient.get<EvidencePackage>(
      `/api/v1/evidence/${chain}/${address}?format=json`
    );
    return res.data;
  },

  downloadEvidenceMarkdown: async (chain: Chain | string, address: string): Promise<string> => {
    const res = await apiClient.get<string>(
      `/api/v1/evidence/${chain}/${address}?format=markdown`,
      { responseType: 'text' }
    );
    return res.data;
  },

  getXGBoostModelInfo: async (): Promise<ModelInfoResponse> => {
    const res = await apiClient.get<ModelInfoResponse>('/api/v1/ml/model-info');
    return res.data;
  },

  getGNNModelInfo: async (): Promise<GNNModelInfoResponse> => {
    const res = await apiClient.get<GNNModelInfoResponse>('/api/v1/gnn/model-info');
    return res.data;
  },

  submitSahyogRequest: async (
    chain: Chain | string,
    address: string,
    action: 'DISCLOSURE' | 'FREEZE_AND_DISCLOSURE' = 'DISCLOSURE'
  ): Promise<SahyogRequest> => {
    const res = await apiClient.post<SahyogRequest>(
      `/api/v1/sahyog/request/${chain}/${address}`,
      { requested_action: action }
    );
    return res.data;
  },

  getSahyogStatus: async (requestId: string): Promise<SahyogRequest> => {
    const res = await apiClient.get<SahyogRequest>(`/api/v1/sahyog/status/${requestId}`);
    return res.data;
  },

  getSahyogRequests: async (): Promise<SahyogRequest[]> => {
    const res = await apiClient.get<SahyogRequest[]>('/api/v1/sahyog/requests');
    return res.data;
  },

  getSahyogVaspRegistry: async (): Promise<{ vasps: VaspRegistryItem[]; disclaimer: string }> => {
    const res = await apiClient.get<{ vasps: VaspRegistryItem[]; disclaimer: string }>('/api/v1/sahyog/vasp-registry');
    return res.data;
  },
};

